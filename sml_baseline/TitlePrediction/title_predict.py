# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/26 19:37
@Author ： Wanglulu
@File ：title_predict.py
@IDE ：PyCharm Community Edition
"""
import codecs
import re
from pypinyin import lazy_pinyin

import unicodedata
import string
import nltk
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
import langid
all_letters = string.ascii_letters
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def zh_sentencesplit(text):
    sentences = re.split('(。|！|\!|\.|？|\?)',text)         # 保留分割符
    new_sents = []
    for i in range(int(len(sentences)/2)):
        sent = sentences[2*i] + sentences[2*i+1]
        new_sents.append(sent)
    return new_sents

#获取一句话是否是过去式
def determine_tense_input(sentence):
    text = word_tokenize(sentence)
    tagged = pos_tag(text)
    past_len= len([word for word in tagged if word[1] in ["VBD", "VBN"]])
    if past_len>0:
        return False
    else:
        return True

#获取名字在文本的存在的比分
def check_name_in_text(name, text):
    """
    Sample: for the name of "Bai Li", \
    # www.xx.com/li.jpg get 0.5
    www.xx.org/bai_li.jpg get 1
    www.xx.org.avatar.jpg get 0
    """
    score = 0
    text = ' '.join(lazy_pinyin(text))
    for i in re.split(r'[\. -]', name):
        if unicodeToAscii(i).lower() in unicodeToAscii(text).lower():
            score += 1
    return score / len(name.split(' '))


#预测职称
class title_predict():
    """
    Guess one's location and position with simple html text
    """

    def __init__(self):
        self.load_data()

    def load_data(self):
        self.pos_list = []
        self.pos_dict={}
        for line in codecs.open('./data/title.txt', 'r', 'utf-8'):
            title_candiate=line.strip().split("\t")[0].lower()
            gold_label=line.strip().split("\t")[1]
            self.pos_list.append(title_candiate)
            self.pos_dict[title_candiate]=gold_label
        self.pos_p = re.compile(r'|'.join([ r'\b'+i+r'\b' for i in self.pos_list]).replace(r'.', r'\.').replace(r'(', r'\(').replace(r')', r'\)'))
        self.year_p = re.compile(r'((?:19|20)[0-9]{2})')

    def check_ta_pos(self, html_text, name, index):
        pos_words_p = re.compile(r'[Pp]osition[^"]|职|[Tt]itle[^=>"]')
        for i in range(len(html_text)):
            if  pos_words_p.search(html_text[i]):
                return True
            if check_name_in_text(name, html_text[i]) > 0.2:
                return True
            for part_name in name.lower().split():
                print (part_name)
                if part_name in html_text[i].lower():
                    return True
        return False
    #判断文本是否对抽取职称有效
    def check_pos(self, html_text, name, index):
        pos_words_p = re.compile(r'[Pp]osition[^"]|职|[Tt]itle[^=>"]')
        for i in range(len(html_text)):
            if  pos_words_p.search(html_text[i]):
                return True
            if check_name_in_text(name, html_text[i]) > 0.2:
                return True
            for part_name in name.lower().split():
                if part_name in html_text[i].lower():
                    return True
        return False
       
    #根据标题和摘要抽取职称
    def pos_guess_ta(self, name, html_text):
        poss = []
        html_text=html_text.split("\n")
        html_text = [i for i in html_text if i != '']
        for i in range(len(html_text)):
            temp_pos = self.pos_p.findall(html_text[i])
            if len(temp_pos) > 0:
                if self.check_pos(html_text, name, i):
                    poss.extend(temp_pos)
        sub_filter = {}
        for i in poss:
            sub_filter[i] = True
        for i in poss:
            for j in poss:
                if i != j and i in j:
                    sub_filter[i] = False
        poss = []
        for i in sub_filter:
            if sub_filter[i]:
                poss.append(self.pos_dict[i])
        return poss
        
    #根据每个页面内的抽取职称
    def pos_guess_html(self, name, html_text):
        poss=[]
        html_text=re.sub("present|至今","2021",html_text)
        html_text=html_text.split("\n")
        html_text = [i for i in html_text if i != '']
        for i in range(len(html_text)):
            temp_pos = self.pos_p.findall(html_text[i])
            if len(temp_pos) > 0:
                if langid.classify(html_text[i])[0]=="zh":
                    sents=zh_sentencesplit(html_text[i])
                else:
                    sents=sent_tokenize(html_text[i])
                for sent in sents:
                    tem_pos = self.pos_p.findall(sent)
                    if len(tem_pos)>0:
                        if determine_tense_input(sent):
                            if self.check_pos(html_text, name, i):
                                poss.extend(tem_pos)
        sub_filter = {}
        for i in poss:
            sub_filter[i] = True
        for i in poss:
            for j in poss:
                if i != j and i in j:
                    sub_filter[i] = False
        poss = []
        for i in sub_filter:
            if sub_filter[i]:
                poss.append(self.pos_dict[i])
        return poss
        