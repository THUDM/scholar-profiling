# import tools
import os
import re
import json
import pandas as pd
from lxml import etree
from tqdm import tqdm

import settings

# search_pages_dir = "C:/Users/YL/Desktop/googleSearch/"
search_pages_dir = "/home/zfj/research-data/user_profiling/googleSearch/"

title_list2Id={ "Other(其他)": 0,
    "B-Professor(教授)": 1,
    "I-Professor(教授)": 2,
    "B-Researcher(研究员)": 3,
    "I-Researcher(研究员)": 4,
    "B-Associate Professor(副教授)": 5,
    "I-Associate Professor(副教授)": 6,
    "B-Assistant Professor(助理教授)": 7,
    "I-Assistant Professor(助理教授)": 8,
    "B-Professorate Senior Engineer(教授级高级工程师)": 9,
    "I-Professorate Senior Engineer(教授级高级工程师)": 10,
    "B-Engineer(工程师)": 11,
    "I-Engineer(工程师)": 12,
    "B-Lecturer(讲师)": 13,
    "I-Lecturer(讲师)": 14,
    "B-Senior Engineer(高级工程师)": 15,
    "I-Senior Engineer(高级工程师)": 16,
    "B-Ph.D(博士生)": 17,
    "I-Ph.D(博士生)": 18,
    "B-Associate Researcher(副研究员)": 19,
    "I-Associate Researcher(副研究员)": 20,
    "B-Assistant Researcher(助理研究员)": 21,
    "I-Assistant Researcher(助理研究员)": 22,
    "B-Student(学生)": 23,
    "I-Student(学生)": 24
    }

def load_data(file): 
    lines = open(file,encoding='utf-8').readlines()
    data=[]
    for l in lines:
        d = json.loads(l)
        data.append(d)
    return data


def r_excel_list(path):
    result=[]
    datas = pd.read_excel(path)
    for i, row in datas.iterrows():
        result.append(dict(row))
    return result

def extract_google_page(file):
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
    file = search_pages_dir + id + '_s1.html'
    # print("file s1 exists", os.path.exists(file))
    if os.path.exists(file):
        link_list = extract_google_page(file)
    # file = '/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/data/' + id+ '_s2.html'
    file = search_pages_dir + id+ '_s2.html'
    if os.path.exists(file):
        link_list.extend(extract_google_page(file))
    # print("link list", link_list)
    return link_list

# 生成性别验证集dev
def get_gender_data(file):
    data = r_excel_list(file)
    json_result = []
    for d in tqdm(data):
        id = d['id']
        name = d['name']
        org = d['org']
        gender = d['gender']
        link_list = get_search_list(id)
        text = name + '[SEP]' + org + '; '  #  + '; '
        if link_list:
            for link in link_list:
                if link['content']:
                    text += link['content'] + '; '
        json_result.append({
            'id': id,
            'name': name,
            'org': org,
            'gender': gender,
            'text': text
        })
    print(len(json_result))
    with open("data/gender_dev.json", 'w', encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=4)

def contain_words(text,words): #判断文本中是否包含一组词语的至少一个
    text = text.lower()
    for word in words:
        word = word.lower()
        if text.find(word)>=0:
            return text
    return ''

# 生成职称验证集dev
def get_title_data(file):
    data = r_excel_list(file)
    json_result = []
    words = { 'Professor', 'Researcher', 'Engineer ','Lecturer', 'Ph.D', 'Research ', 'Student'}
    for d in tqdm(data):
        id = d['id']
        name = d['name']
        org = d['org']
        title = d['title']
        link_list = get_search_list(id)
        text = name + '; ' + org + '; '
        if link_list:
            for link in link_list:
                if link['content']:
                    word = contain_words(link['content'], words)
                    text += word + '; '
        text = text.replace('\n', ' ').replace('\t', ' ')
        json_result.append({
            'id': id,
            'name': name,
            'org': org,
            'title': title,
            'text': text
        })

    print(len(json_result))
    with open("data/dev.json", 'w', encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=4)

# 生成测试集（性别、职称）test
def get_gender_test():
    data = r_excel_list(r'data/raw/new_test.xlsx')
    result = {}
    json_result = []
    words = {'Professor', 'Researcher', 'Engineer ', 'Lecturer', 'Ph.D', 'Research ', 'Student'}
    for d in tqdm(data):
        id = d['id']
        result[id] = d
        result[id]['id'] = id
        result[id]['name'] = d['name']
        result[id]['org'] = d['org']
        result[id]['gender'] = ''
        result[id]['title'] = ''
        link_list = get_search_list(id)
        result[id]['gender_text'] = d['name'] + '[SEP]' + d['org'] + '; '  # + '; '
        result[id]['title_text'] = d['name'] + '; ' + d['org'] + '; '
        if link_list:
            for link in link_list:
                if link['content']:
                    result[id]['gender_text'] += link['content'] + '; '
                    word = contain_words(link['content'], words)
                    result[id]['title_text'] += word + '; '

        result[id]['title_text'] = result[id]['title_text'].replace('\n', ' ').replace('\t', ' ')

    for key in ['gender', 'title']:  # ['homepage','lang','gender','title']
        homepage_data = load_data('data/raw/ground_truth.json')
        for d in homepage_data:
            result[d['id']][key] = d[key]

    for id in result:
        title_data = {"id": result[id]['id'], "name": result[id]['name'], "org": result[id]['org'], "gender": result[id]['gender'], "title": result[id]['title'],
                      "gender_text": result[id]['gender_text'], "title_text": result[id]['title_text']}
        json_result.append(title_data)
    with open("data/test.json", 'w', encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=4)

# 生成训练数据（性别、职称）train
def get_train_data():
    json_result = []
    data = r_excel_list(r'data/raw/new_train.xlsx')
    words = {'Professor', 'Researcher', 'Engineer ', 'Lecturer', 'Ph.D', 'Research ', 'Student'}
    for d in tqdm(data):
        id = d['id']
        name = d['name']
        org = d['org']
        gender=d['gender']
        title = d['title']
        link_list = get_search_list(id)
        gender_text = name+'[SEP]'+org
        title_text = name + '; ' + org + '; '
        if link_list:
            for link in link_list:
                if link['content']:
                    gender_text += link['content'] + '; '
                    t = contain_words(link['content'], words)
                    title_text += t + '; '

        gender_text = gender_text[:240].replace("\n", " ").replace("\r", " ").replace("\t", " ")
        title_text = title_text.replace('\n', ' ').replace('\t', ' ')

        title_data = {"id": id, "name": name, "org": org, "gender": gender, "title": title,
                      "gender_text": gender_text, "title_text": title_text}
        json_result.append(title_data)
    print(len(json_result))
    with open("data/train.json", 'w', encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=4)

def find_head_idx(source, target):
    head_idx=[]
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            head_idx.append(i)
    return head_idx

# 生成职称训练数据（用于序列标注）(train dev test)
def get_title_tag(tokenizer = None):
    '''
    生成职称文本对应序列标注
    :param tokenizer:
    :return:
    '''

    train_data = json.load(open('data/dev.json')) # 使用上方处理后的json文件
    title2tis = json.load(open(r'data/title2tis.json'))

    # for data in train_data:
    # Researcher(研究员) 和 Research(研究员) 合并
    result = []
    title_list = ["Professor(教授)","Ph.D(博士生)","Associate Professor(副教授)","Assistant Professor(助理教授)",
                  "Engineer(工程师)","Senior Engineer(高级工程师)","Professorate Senior Engineer(教授级高级工程师)","Lecturer(讲师)",
                  "Researcher(研究员)","Associate Researcher(副研究员)","Assistant Researcher(助理研究员)","Student(学生)"]
    total={}
    for data in tqdm(train_data):
        # text_lists = []
        # text_tags = []
        text_list = re.sub('[\u4e00-\u9fa5]', '', data['text']).split("; ") # 去除字符串之中的中文
        # text_list = data['text'].split("; ")
        for text in text_list:
            if text != '':
                text_tokens = tokenizer.tokenize(text)
                if len(text_tokens) < 10: continue # 过滤太短的句子
                # text_lists.append(text)
                label = [0] * len(text_tokens)
                if data['title'] != "Other(其他)":
                    for ti_num in title_list:
                        for tis in title2tis[ti_num]:
                            title = tokenizer.tokenize(tis)
                            title_head_idx = find_head_idx(source=text_tokens, target=title)  # 返回主语在text_tokens第一个位置的下标
                            if len(title_head_idx) > 0:
                                for idx in title_head_idx:
                                    label[idx] = title_list2Id[r'B-' + ti_num]
                                    if len(title) > 1:
                                        label[idx + 1:idx + len(title)] = (title_list2Id[r'I-' + ti_num],) * (len(title) - 1)
                    if label == [0] * len(text_tokens):
                        continue
                title_data = {"id": data["id"], "title": data["title"], "text": text, "text_tags": label}
                result.append(title_data)
    print(len(result))
    with open("data/title_dev.json", 'w', encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    # get_gender_data(r'data/raw/new_dev.xlsx')
    # get_title_data(r'data/raw/new_dev.xlsx')
    # get_gender_test()
    # get_train_data()

    from pathlib import Path
    from transformers import BertTokenizer
    root_path = Path(os.path.abspath(os.path.dirname(__file__)))
    bert_model_dir = root_path / 'pretrain_models/bert_base_cased'
    tokenizer = BertTokenizer(vocab_file=os.path.join(bert_model_dir, 'vocab.txt'), do_lower_case=True)  # 大小写不敏感
    get_title_tag(tokenizer)
