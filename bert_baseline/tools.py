import json
import os
import pandas as pd
from tqdm import tqdm
import random
import urllib.parse

import settings


def load_data(file): #加载数据
    lines = open(file,encoding='utf-8').readlines()
    data=[]
    for l in lines:
        d = json.loads(l)
        data.append(d)
    return data


#读取excel数据集 转换为相应数组
def r_excel_list(path):
    result=[]
    datas = pd.read_excel(path)
    for i, row in datas.iterrows():
        result.append(dict(row))
    return result

# train_path="/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_train.xlsx"
# test_path="/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_test.xlsx"
# train_path="../../data/new_train.xlsx"
train_path = os.path.join(settings.DATA_DIR, "raw", "new_train.xlsx")
# test_path="../../data/new_test.xlsx"
test_path = os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx")

# search_pages_dir = "/home/zfj/research-data/user_profiling/googleSearch/"
search_pages_dir = os.path.join(settings.DATA_DIR, "googleSearch", "data")
bert_input_dir = os.path.join(settings.DATA_DIR, "bert_input")
os.makedirs(bert_input_dir, exist_ok=True)

train_data = r_excel_list(train_path)
valid_data=r_excel_list(test_path)

print(len(train_data))
# valid_data = load_data (r'C:\Users\PC\PycharmProjects\CCKS\CCKS文档\学者画像\dev_for_test.json')

# result ={}
# count=0 #统计数据
# for d in train_data:
#     lang =  d['gender']
#     if lang in result:
#         result[lang]+=1
#     else:
#         result[lang]=1
# for k, v in result.items():
#     print(k, (v/len(train_data)))

from lxml import etree


def extract_google_page(file):#加载google搜索列表页
    html = open(file,encoding='utf-8').read()
    html = etree.HTML(html)
    link_list =[]
    items = html.xpath('//div[@class="ZINbbc xpd O9g5cc uUPGi"]')
    for item in items:
        try:
            href = item.xpath('.//a')[0].attrib['href']
            start = href.find('http')
            end = href.find('&sa')
            href = href[start:end]
            title = item.xpath('.//div[@class="BNeawe vvjwJb AP7Wnd"]')[0].text
            content = item.xpath('.//div[@class="BNeawe s3v9rd AP7Wnd"]')[1].text
            link ={'href':href,'title':title,'content':content}
            link_list.append(link)
        except:
            pass
    return link_list

#加载给定id的两个搜索列表页，结果合并在一起
def get_search_list(id):
    link_list = []
    # file = '/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/data/' + id+ '_s1.html'
    # file = search_pages_dir + id + '_s1.html'
    file = os.path.join(search_pages_dir, id+"_s1.html")
    # print("file s1 exists", os.path.exists(file))
    if os.path.exists(file):
        link_list = extract_google_page(file)
    # file = '/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/data/' + id+ '_s2.html'
    # file = search_pages_dir + id+ '_s2.html'
    file = os.path.join(search_pages_dir, id+"_s2.html")
    if os.path.exists(file):
        link_list.extend(extract_google_page(file))
    # print("link list", link_list)
    return link_list

#加载给定id的所有内容页，返回一个列表，列表元素为每个内容页的html文本
def get_content_pages(id):
    contents = []
    for i in range(20):
        # file = '/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/data/' + id + '_' + str(i) + '.html'
        file = search_pages_dir + id + '_' + str(i) + '.html'
        if os.path.exists(file):
            try:
                html = open(file,encoding='utf-8').read()
                contents.append(html)
            except Exception:
                pass
                # print('load html error')
    return contents


def contain_words(text,words): #判断文本中是否包含一组词语的至少一个
    text = text.lower()
    for word in words:
        word = word.lower()
        if text.find(word)>=0:
            return text
    return ''

lang2id={}
id2lang={}
def create_language_classification_data():    #生成“语言分类”训练数据
    global  lang2id
    global  id2lang
    outfile =open('lang.train','w',encoding='utf-8')
    langs = []
    for d in train_data:
        lang = d['lang']
        if lang not in langs:
            langs.append(lang)
    lang2id = {langs[i]: i for i in range(len(langs))}
    id2lang = {i: langs[i] for i in range(len(langs))}
    words =['chinese','china','american','USA','US','india','Holland','Netherlands','italy','japan','france'
             ,'spain','greece','germany','korea','russian','portugal','sweden','Bangladesh',
             'english', 'chinese', 'indian', 'dutch',  'italian', 'japanese',  'french',
             'spanish', 'greek',  'german', 'korean', 'russian', 'portuguese', 'arabic',
             'unknown', 'swedish', 'bengali']   #关键词列表
    for d in train_data:
        name = d['name']
        org = d['org']
        lang = d['lang']
        id = lang2id[lang]
        # if id == 1:continue  #'中文'不再训练(使用规则提取)
        text = name + '; ' + org + '; '
        link_list = get_search_list(d['id'])
        if link_list:
            for link in link_list:
                if link['content']:
                    t = contain_words(link['content'], words)
                    if t!='':
                        text += t + '; '
                        break
        text = text.replace('\n',' ').replace('\t',' ')
        text += '\t'+ str(id) + '\n'
        outfile.write(text)
    outfile.close()
# create_language_classification_data()
# print(id2lang)

def is_same_url(url1, url2):
    if url1.find(url2) >= 0:
        return 1
    if url2.find(url1) >= 0:
        return 1
    return 0
def clear_url(url):
    url =url.replace('http://','')
    url =url.replace('https://','')
    url =url.replace('www.','')
    return url
def get_domain(url):
    index = url.index('/')
    return url[:index]

pos_domains = set()
neg_domains = set()
def create_homepage_classification_data(): #生成“主页分类”训练数据
    # outfile = open('homepage.train','w',encoding='utf-8')
    outfile = open(os.path.join(bert_input_dir, "homepage.train"),'w',encoding='utf-8')
    for d in tqdm(train_data):
        id = d['id']
        name = d['name']
        org = d['org']
        homepages = eval(d['homepage'])
        link_list = get_search_list(id)
        if not link_list:continue
        hrefs =[]
        for link in link_list:
            href = link['href']
            href = urllib.parse.unquote(href)
            hrefs.append(href)
            href = clear_url(href)
            y = 0
            for h in homepages:
                if href.find(h)>=0:
                    y=1
                    break
                if h.find(href)>=0:
                    y=1
                    break
            # if link['content']:
            #     text += link['content']+'; '
            # text = text.replace('\n',' ').replace('\t',' ')
            # text += '\t' + str(y) + '\n'

            text = name+ '[SEP]' + href + '\t' + str(y) + '\n'
            domain = get_domain(href)
            if y==1:
                pos_domains.add(domain)
                outfile.write(text)
            elif random.random() < 0.3:
                neg_domains.add(domain)
                outfile.write(text)
    outfile.close()
# create_homepage_classification_data()



#
def create_gender_classification_data(): #生成“主页分类”训练数据
    # outfile =open('gender.train','w',encoding='utf-8')
    outfile = open(os.path.join(bert_input_dir, "gender.train"),'w',encoding='utf-8')
    for d in tqdm(train_data):
        id = d['id']
        name = d['name']
        org = d['org']
        gender=d['gender']
        link_list = get_search_list(id)
        text=name+'[SEP]'+org
        if  link_list:
            for link in link_list:
                if link['content']:
                    text += link['content'] + '; '
        if gender=="female":
            y=0
        else:
            y=1
        text = text[:240].replace("\n", " ").replace("\r", " ").replace("\t", " ") + '\t' + str(y) + '\n'
        outfile.write(text)
    outfile.close()
# create_gender_classification_data()
# union = neg_domains & pos_domains
# neg_domains = neg_domains - union
# pos_domains = pos_domains - union

title2id={}
id2title={}
titleCount={}
def create_title_classification_data():
    global  title2id
    global  id2title
    outfile =open('title.train','w',encoding='utf-8')
    titles = []
    for d in tqdm(train_data):
        title = d['title']
        if title not in titles:
            titles.append(title)
        if title not in titleCount:
            titleCount[title]=0
        titleCount[title]+=1
    title2id = {titles[i]: i for i in range(len(titles))}
    id2title = {i: titles[i] for i in range(len(titles))}

    words = { 'Professor', 'Researcher', 'Engineer ','Lecturer', 'Ph.D', 'Research ', 'Student'}
    # words ={'Professorate Senior Engineer','Senior Engineer','Associate Researcher','Assistant Researcher','Assistant Professor'
    #         ,'Associate Professor','Researcher ','Professor ','Lecturer ','Ph.D',' Student '}
    for d in train_data:
        id = d['id']
        name = d['name']
        org = d['org']
        title = d['title']
        link_list = get_search_list(id)
        text = name + ';' + org + '; '
        if not link_list:continue
        for link in link_list:
            if link['content']:
                t = contain_words(link['content'],words)
                text += t + '; '
        text = text.replace('\n',' ').replace('\t',' ')
        id = title2id[title]
        text += '\t'+ str(id) + '\n'
        outfile.write(text)
    outfile.close()
# create_title_classification_data()
# print(id2title)
#
def merge_result():
    # data = r_excel_list(r'/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_test.xlsx')
    # data = r_excel_list(r'../../data/new_test.xlsx')
    data = r_excel_list(os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx"))
    result = {}
    for d in data:
        id = d['id']
        result[id] = d
        result[id]['homepage']=[]
        result[id]['title'] = 'Professor(教授)'
        result[id]['gender'] = 'male'
    for key in ['gender','homepage','title']:          #['homepage','lang','gender','title']
        homepage_data = load_data(r'../../data/'+'new_test-'+key+'.json')
        for d in homepage_data:
            result[d['id']][key] = d[key]
    # outf = open(r'../../data/luoyang-result_new.json', 'w', encoding='utf-8')
    outf = open(os.path.join(settings.DATA_DIR, "luoyang-result_new.json"), 'w', encoding='utf-8')
    # count=0
    for id in result:
        homepage = result[id]['homepage']  # 主页
        homepage = [i for i in homepage if i.find('scholar.google') == -1 and i.find('dblp') == -1]  # 过滤主页
        result[id]['homepage'] = homepage
        if result[id]['name'] == '':          #清理数据,提高(74.53-->74.57)
            result[id]['homepage'] = []
            result[id]['title'] = ''
            result[id]['gender'] = ''
        s = json.dumps(result[id],
            ensure_ascii=False)
        outf.write(s + '\n')
    outf.close()
'''
各项分别使用male、english、professor以及默认空值，就可以获得47.9分
gender：样本男女比例约为9：1,使用女性代词（she her 等）做为区别性特征(有轻微的提高)
title：教授约占70%，使用神经网络模型可以达到81%,总分提高约3个点
email:采用正则表达式匹配邮箱地址，然后计算与人名的相似度，并进行筛选，总分提高约4个点 
homepage：使用神经网络模型（大部分都不是主页，所以用F1值衡量，结果约74%）。
lang：用神经网络模型，使总分提高约5个点(47.5 -> 56)（使用中文规则+神经网络47.5 -> 56.07）。
使用上述结果提交Test1，得分约为63.1
2121.7.9:使用规则、统计、机器学习的方法在非aminer数据上得分为42.4（总分约66.4）
         使用网页信息抽取的方法在aminer数据上得分为30.1（总分为33.6）
         
202.7.16：采用规则、统计、机器学习的方法，不使用aminer数据，总分为64.44，使用aminer后，部分为75.78
'''

# create_gender_classification_data()
create_homepage_classification_data()
# create_title_classification_data()
# merge_result()
