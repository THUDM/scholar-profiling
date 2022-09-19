#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : homepage_train.py
# @Author: KaiYu
# @Date  : 2022/4/14
# @Desc  : homepage训练（xgboost）
import xgboost as xgb
import spide
import os
import json
import pandas as pd
import html_text
from collections import defaultdict
from tqdm import tqdm
import chardet
import re
import numpy as np
from pypinyin import lazy_pinyin
import time
import datetime
import warnings
from collections import defaultdict as dd
import requests
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer  #原始文本转化为tf-idf的特征矩阵
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
# from sklearn.random_projection import sparse_random_matrix
import pickle
import argparse
from scipy import sparse

import settings

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', type=str, default="xgboost", help='classifier')
args = parser.parse_args()

#读取excel数据集 转换为相应数组
def r_excel_list(path):
    result=[]
    datas = pd.read_excel(path)
    for i, row in datas.iterrows():
        result.append(dict(row))
    return result

# train_data="/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_train.xlsx"
# dev_data="/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_dev.xlsx"
# test_data="/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_test.xlsx"
# train_data="../../data/new_train.xlsx"
# dev_data="../../data/new_dev.xlsx"
# test_data="../../data/new_test.xlsx"
train_data = os.path.join(settings.DATA_DIR, "raw", "new_train.xlsx")
dev_data = os.path.join(settings.DATA_DIR, "raw", "new_dev.xlsx")
test_data = os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx")

train_data=r_excel_list(train_data)
test_data=r_excel_list(test_data)


warnings.filterwarnings('ignore')
homepage_pos = [u'edu', u'faculty', u'id', u'staff',  u'detail', u'person', u'about', u'academic', u'teacher', u'list', u'people', \
                u'lish', u'homepages', u'researcher', u'team', u'teachers', u'member', u'profile']
homepage_neg = [u'books', u'google', u'pdf', u'esc', u'scholar', u'netprofile', u'linkedin', u'researchgate', u'news',\
                u'article', u'[^n]wikipedia', u'gov', u'showrating', u'youtube', u'blots', u'citation', u'expert', \
                u'dblp', u'researchgate', u'baidu', u'aminer', u'irps', u'taobao']
search_file = open('search_log.json', 'a', encoding='utf-8')
log_file = open('homepage.log', 'a', encoding='utf-8')


def check_name_in_text(name, text):
    """
    Sample: for the name of "Bai Li", \
    # www.xx.com/li.jpg get 0.5
    www.xx.org/bai_li.jpg get 1
    www.xx.org.avatar.jpg get 0
    """
    score = 0
    if name != None:
        # text = ' '.join(lazy_pinyin(text))
        for i in re.split(r'[ \-,.]', name):
            if i.lower() in text.lower():
                score += 1
        return score / len(name.split(' '))
    else:
        return 0


def one_sample_homepage_features(data, search_res, labeled=True, rank=None):
    """
    Input: data - the row of dataframe. search_res - the list of search results info
    Ouput: features
    """
    features = []
    p = re.compile(r'University|university|大学|institute|School|school|college|学院')
    pos_p = re.compile('|'.join(homepage_pos))
    neg_p = re.compile('|'.join(homepage_neg))
    name = data['name'].replace('?', '')
    organize = data['org']
    if data["org"] != None and p.match(data["org"]):
        in_school = 1  # 2
    else:
        in_school = 0

    if search_res == None:
        return []
    for i in range(len(search_res)):
        title = search_res[i][0]
        url = search_res[i][1]
        if rank == None:
            rank = [i]  # 1
        else:
            rank = [rank]
        if labeled:
            if url == data.homepage:
                label = 1
            else:
                # subsample
                # if random.random() < 0.3:
                # if rank < 2:
                label = 0
        else:
            # if rank >= 2:
            # continue
            label = url
        feature = []
        feature.append(label)
        feature.extend(rank)
        content = search_res[i][2]
        # is_cited = search_res[i][3] # 3
        pos_words_num = len(pos_p.findall(url.lower()))  # 4
        pos_title = len(pos_p.findall(title.lower()))
        neg_words_num = len(neg_p.findall(url.lower()))  # 5
        neg_title = len(neg_p.findall(title.lower()))  # 5
        edu = 1 if 'edu' in url else 0  # 6
        org = 1 if 'org' in url else 0  # 7
        gov = 1 if 'gov' in url else 0  # 8
        # name_in = 1 if len(name_p.findall(title.lower())) != 0 else 0 # 9
        linkedin = 1 if 'linkedin' in url else 0  # 10
        google = 1 if 'google' in url else 0  # 10
        gate = 0 if 'researchgate' in url else 1
        title_len = len(title)  # 11
        url_in = check_name_in_text(' '.join(lazy_pinyin(name)), url)
        if organize != None:
            url_org_in = check_name_in_text(' '.join(lazy_pinyin(organize)), url)
        else:
            url_org_in = 0

        if content != None:
            content = ' '.join(content.split('\n'))
            content_len = len(content)  # 12
            name_content = check_name_in_text(name, content)  # 15
            name_count = len(re.findall(name, content))
            #             pos_content = len(pos_p.findall(content.lower()))
            #             neg_content = len(neg_p.findall(content.lower()))  # 5
            if organize != None:
                org_content = check_name_in_text(organize, content)
            else:
                org_content = 0
        else:
            content_len = 0
            name_content = 0
            name_count = 0
            org_content = 0
        #             pos_content = 0
        #             neg_content = 0

        name_title = check_name_in_text(name, title)  # 14
        if organize != None:
            org_title = check_name_in_text(organize, title)  # 14
        else:
            org_title = 0
        # mail_content = 1 if 'mail' in content.lower() else 0 # 16
        # address_content = 1 if 'address' in content.lower() else 0 # 17
        # feature.extend([in_school, is_cited, pos_words_num, neg_words_num, edu,\
        #  org, gov, linkedin, title_len, content_len, org_len, name_title, name_content])
        # linkedin, gate, google, content_len,
        feature.extend([content_len, title_len, pos_words_num, neg_words_num, google, pos_title, neg_title,
                        edu, url_in, url_org_in, org, gov, name_title, name_content, in_school, name_count,
                        org_title, org_content])

        features.append(feature)
    return features


def check_homepage(pre, reference):
    grams_reference = set(reference)  # 去重；如果不需要就改为list
    temp = False
    for i in grams_reference:
        i = i.strip("https://").strip("http://")
        j = pre.strip("https://").strip("http://")
        if j == i:
            temp = True
            break
        elif j.startswith(i):
            temp = True
            break
        elif i.startswith(j) and len(i) - (len(j)) < 10:
            temp = True
            break
        else:
            pass
    return temp

# search_pages_dir = "/home/zfj/research-data/user_profiling/googleSearch/"
search_pages_dir = os.path.join(settings.DATA_DIR, "googleSearch", "data")

train_item = defaultdict(list)
train_labels = []
train_features = []
train_content = []
train_urls = []
for train in tqdm(train_data):
    item_id = train['id']
    name = train['name']
    org = train['org']
    train_urls.append(train['homepage'])
    res = []
    single_item = []
    search_file1 = item_id + '_s1.html'
    search_file2 = item_id + '_s2.html'
    if os.path.exists(os.path.join(search_pages_dir, search_file1)):
        f1 = open(os.path.join(search_pages_dir, search_file1), 'rb')
        f1_content = ''.join([str(line) for line in f1.readlines()])
        f1.close()
        res.extend(spide.extra_deep_url(f1_content))
    else:
        # fout.write(json.dumps({item_id: []}) + '\n')
        # print(item_id, [], train['homepage'])
        print(item_id)

    if os.path.exists(os.path.join(search_pages_dir, search_file2)):
        f2 = open(os.path.join(search_pages_dir, search_file2), 'rb')
        f2_content = ''.join([str(line) for line in f2.readlines()])
        f2.close()
        res.extend(spide.extra_deep_url(f2_content))

    try:
        i = 0
        neg_url = ['researchgate', 'linkedin', 'ieee', 'vt.academia.edu', 'bloomberg']
        skip = False
        url = []
        for item in res:
            for uu in neg_url:
                if uu in item['url']:
                    skip = True
                    break
            if skip:
                continue
            #             if 'edu' not in item['url']:
            #                 continue
            url.append(item['url'])
            con = item['title'] + ' ' + item['text']
            feature = one_sample_homepage_features({'name': name, 'org': org},
                                                   [[item['title'], item['url'], con]],
                                                   labeled=False, rank=i + 1)
            label = check_homepage(item['url'], train['homepage'])
            #             if label == False and random.random() < 0.5:
            #                 i += 1
            #                 continue
            train_labels.append(label)
            train_features.append(feature[0][1:])
            train_content.append(con)

            i += 1
        # aout.write(json.dumps({item_id: url}) + '\n')
    except Exception as e:
        print(e)


stopwords = set()
with open(os.path.join(settings.DATA_DIR, 'stopword.txt'), encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())

train_content_new = []
for sentence in tqdm(train_content):
    s = jieba.lcut(sentence)
    s = [w.lower() for w in s if w not in stopwords and not w.isdigit()]
    train_content_new.append(' '.join(s))

vectorizer=TfidfVectorizer(lowercase=True)  #定义了一个类的实例
X=vectorizer.fit_transform(train_content_new)

with open('vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)

with open('vectorizer.pk', 'rb') as fin:
    vectorizer = pickle.load(fin)


train_item = defaultdict(list)
test_labels = []
test_features = []
test_content = []
test_urls = []
test_pairs = []
for train in tqdm(test_data):
    item_id = train['id']
    name = train['name']
    org = train['org']
    test_urls.append(train['homepage'])
    res = []
    single_item = []
    search_file1 = item_id + '_s1.html'
    search_file2 = item_id + '_s2.html'
    if os.path.exists(os.path.join(search_pages_dir, search_file1)):
        f1 = open(os.path.join(search_pages_dir, search_file1), 'rb')
        f1_content = ''.join([str(line) for line in f1.readlines()])
        f1.close()
        res.extend(spide.extra_deep_url(f1_content))
    else:
        # fout.write(json.dumps({item_id: []}) + '\n')
        # print(item_id, [], train['homepage'])
        print(item_id)

    if os.path.exists(os.path.join(search_pages_dir, search_file2)):
        f2 = open(os.path.join(search_pages_dir, search_file2), 'rb')
        f2_content = ''.join([str(line) for line in f2.readlines()])
        f2.close()
        res.extend(spide.extra_deep_url(f2_content))

    try:
        i = 0
        neg_url = ['researchgate', 'linkedin', 'ieee', 'vt.academia.edu', 'bloomberg']
        skip = False
        url = []
        # print("res", res)
        for item in res:
            # print("here")
            for uu in neg_url:
                # print("url", item["url"])
                if uu in item['url']:
                    skip = True
                    break
            if skip:
                continue
#             if 'edu' not in item['url']:
#                 continue
            url.append(item['url'])
            con = item['title'] + ' ' + item['text']
            # print("here1")
            test_pairs.append((train["id"], item["url"]))
            feature = one_sample_homepage_features({'name': name, 'org': org},
                                                   [[item['title'], item['url'], con]],
                                                   labeled=False, rank=i +1)
            # print("here2")
            # print(item["url"], train["homepage"])
            # print(check_homepage(item["url"], train["homepage"]))
            # test_labels.append(check_homepage(item['url'], train['homepage']))
            test_labels.append(False)
            # print("here3", feature[0])
            test_features.append(feature[0][1:])
            print("here4")
            test_content.append(con)
            # print("con", con)
            i += 1
        # aout.write(json.dumps({item_id: url}) + '\n')
    except Exception as e:
        print(e)

assert len(test_features) == len(test_pairs)

test_content_new = []
for sentence in tqdm(test_content):
    s = jieba.lcut(sentence)
    s = [w.lower() for w in s if w not in stopwords and not w.isdigit()]
    test_content_new.append(' '.join(s))
test_X = vectorizer.transform(test_content_new)  #定义了一个类的实例


train_fea = np.array(train_features)
train_label = np.array(train_labels)
test_fea = np.array(test_features)
test_label = np.array(test_labels)
train_features = sparse.hstack([train_fea, X])
test_features = sparse.hstack([test_fea, test_X])

if args.classifier == 'lr':
    clf = LogisticRegression()
    clf.fit(train_features, train_label)
    pred_label = clf.predict(test_features)
elif args.classifier == 'xgboost':
    xgb1 = xgb.XGBClassifier(
    learning_rate =0.1,
    n_estimators=200,
    max_depth=5,
    min_child_weight=3,
    gamma=0.0,
    subsample=0.5,
    colsample_bytree=0.9,
    objective= 'binary:logistic',
    reg_alpha=0.1,
    scale_pos_weight=1,
    random_state=22)
    xgb1.fit(train_features, train_label)
    pred_label = xgb1.predict(test_features)
else:
    raise NotImplementedError
# xgb1.save_model('new.model')

aid_to_url = dd(set)
for i in range(len(test_pairs)):
    aid, url = test_pairs[i]
    cur_pred = pred_label[i]
    if cur_pred:
        aid_to_url[aid].add(url)

# df_test=pd.read_excel("../../data/new_test.xlsx",keep_default_na=False)
df_test = pd.read_excel(os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx"), keep_default_na=False)

os.makedirs("output/sml/", exist_ok=True)
fw=open("output/sml/homepage_predict_{}.json".format(args.classifier),'w',encoding="utf-8")
for i,row in df_test.iterrows():
    id=row["id"]
    name=row["name"]
    org=row["org"]
    cur_hp = list(aid_to_url.get(id, set()))
    predict_data={"id":id,"name":name,"org":org,"homepage": cur_hp}
    title_data=json.dumps(predict_data)
    fw.write(title_data+"\n")
fw.close()
