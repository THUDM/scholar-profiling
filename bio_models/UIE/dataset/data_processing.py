from audioop import avg
from cgi import print_form
from importlib import import_module
import logging
import os
from tqdm import tqdm
from bs4 import BeautifulSoup 
import pandas as pd
import json
import re
import random

# 进保留以下类别
ch2en={"任职机构": 'work_for', # 现在正在工作的单位
    "职称": 'title', # 目前的职称或者职位
    "性别": 'gender', # 标注一次即可
    "出生地": 'birth_place',
    "出生日期": 'birthday',
    "最高学历": 'highest_education',
    "研究方向": 'research_interests',
    "荣誉称号": 'honorary_title',
    "获得奖项": 'awards',
    "教育信息": 'education',
    "工作履历": 'work_record',
    "社会任职": 'take_office',
    }

# 批量ANSI文件转UTF-8 同时将中文标签转为英文
def convert_dir_to_utf8(file_path, target_path):
    files = os.listdir(file_path)
    for file in files:
        file_name = file_path + '/' + file
        f = open(file_name, 'r', encoding='gbk')
        ff = f.read()
        for old_str, new_str in ch2en.items():
            ff = ff.replace(old_str, new_str)
        file_object = open(target_path + '/' + file, 'w', encoding='utf-8')
        file_object.write(ff)

# 返回第idx个匹配的索引
def find_head_idx(source, target, idx=1):
    """从sequence中寻找第idx个子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(target)
    f = 0
    for i in range(len(source)):
        if source[i:i + n] == target:
            f += 1
            if f == idx:
                return i
    return -1

# 初步处理 并划分数据集(sample一些文件作为验证机和测试集)
def sample_file(file_path,new_file_path,ratio):
    pattern = re.compile('(?<! )(?=[\-\*\[\]\/.,;:`\'\"!?()])|(?<=[\-\*\[\]\/.,;:`\'\"!?()])(?! )') # [] * - \
    files = os.listdir(file_path)
    # files = ['en_bio_18.sgml', 'en_bio_0.sgml', 'en_bio_20.sgml', 'en_bio_24.sgml', 'en_bio_31.sgml', 'en_bio_45.sgml', 'en_bio_30.sgml']

    outf = open(new_file_path + '/' + 'en_bio.json', 'w', encoding='utf-8')
    train_outf = open(new_file_path + '/' + 'en_bio_train.json', "w", encoding="utf-8")
    val_outf = open(new_file_path + '/' + 'en_bio_val.json', "w", encoding="utf-8")
    test_outf = open(new_file_path + '/' + 'en_bio_test.json', "w", encoding="utf-8")
    trainlist, vallist, testlist = [], [], []
    num = 0
    for file in files:
        
        file_name = file_path + '/' + file
        print(("Reading lines from {}".format(file_name)))

        # 随机分配文件 生成训练集 验证集 测试集
        if random.random() > ratio:
            if file not in trainlist:
                trainlist.append(file)
            outf_ = train_outf
        elif random.random() > 0.5:
            if file not in vallist:
                vallist.append(file)
            outf_ = val_outf
        else:
            if file not in testlist:
                testlist.append(file)
            outf_ = test_outf
        
        with open(file_name, "r", encoding="utf-8") as f:
            data = f.read()
            soup = BeautifulSoup(data, 'html.parser')
            results = soup.find_all('entry')
            for item in tqdm(results):
                ner = []
                docid = item.find("docid").text
                raw_text = item.find("text") # .strip()
                text = raw_text.text
                # print(text)
                for label in ch2en.values():
                    entities = raw_text.find_all(label)
                    text_ = re.sub(pattern, r' ', text).split()
                    # print(text_.split())
                    en = {}
                    for entity in entities:
                        # entity_= entity.text.split()
                        if entity.text == "": # 会有实体为空的现象 还有重复实体
                            # print("entity", entity.text)
                            continue
                        entity_ = re.sub(pattern, r' ', entity.text).split()
                        if entity.text not in en:
                            en[entity.text] = 1
                        else: en[entity.text] +=1
                        head_idx = find_head_idx(text_, entity_, en[entity.text])
                        if head_idx != -1:
                            num += 1
                            ner.append([head_idx, head_idx+len(entity_), label])

                        # 输出有问题的数据
                        if head_idx < 0:
                            print(text_)
                            print(entity)
                            print(entity_)
                            print([head_idx, head_idx+len(entity_), label])
                
                s = json.dumps({
                'docid':docid,
                'text': " ".join(text_), # text.strip('\n'),
                'tokens':text_,
                'ner':ner
            },
                ensure_ascii=False)
                outf.write(s + '\n')
                outf_.write(s + '\n')
            
    print(num)
    outf.close()
    train_outf.close()
    val_outf.close()
    test_outf.close()
    print("finished...")
    print(f"trainlist: {trainlist}\n vallist: {vallist}\n testlist: {testlist}\n")


if __name__ == '__main__':

    # 批量转换格式 防止乱码
    # 将文件中的中文标签替换为英文 方便后续处理
    file_path = "/shenyelin/UIE-main/dataset/en_labeled"
    target_path = "/shenyelin/UIE-main/dataset/raw_labeled"
    convert_dir_to_utf8(file_path, target_path)
    
    # 设置随机种子 
    seed = 12
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)

    file_name="/shenyelin/bio_baselines/UIE-main/dataset/raw_labeled" # 'train.sgml'
    new_file_name = "/shenyelin/bio_baselines/UIE-main/dataset/raw_en_bio" # en_bio

    # 随机sample一些文件作为验证集和测试集
    sample_file(file_name,new_file_name,0.3)
    

    '''
    标注问题较多 已人工校对
    <work_for>Texas Southwestern Medical Schoo</work_for>l
    <title>Senior Software Engineer</title><work_for>Google, Inc.</work_for>

    生成无法完全一致
    <highest_education>doctor</highest_education>s (17)
    <awards>Robert Wood Johnson Foundation Minority Faculty Development Award</awards>ee
    数据后可能会有引用标记
    '''