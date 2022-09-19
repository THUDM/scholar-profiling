# -*- coding: utf-8 -*-
"""
@Time ： 2021/4/14 9:32
@Author ： Wanglulu
@File ：generate_features.py
@IDE ：PyCharm Community Edition
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import unicodedata
import string
import syllables
import nltk

#转码
all_letters = string.ascii_letters
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
#切词
def ngram_split(word,n):
    chars=[c for c in word]
    ngram=nltk.ngrams(chars,n)
    subword= [u''.join(w) for w in ngram]
    if subword!=[]:
        return subword
    else:
        return [word]

#提取名字特征
def extract_name_features(name):
    #name=name.split()
    try:
        first=name[0]
        last_two=name[-2:]
        last_three=name[-3:] if len(name) > 2 else name
        last_four=name[-4:] if len(name) > 3 else name
        last_five=name[-5:] if len(name) > 4 else name
        last_six=name[-6:] if len(name) > 5 else name
        return [first,last_two,last_three,last_four,last_five,last_six]
    except:
        return ["#","#","#","#","#","#"]

#获取名字词长、字符长、音节、以及2-gram，3-gram长度
def get_length_feature(name):
    try:
        word_len=len(name.split())
        char_len=len(name)
        syl_len=syllables.estimate(name)
        bigram_len=len(ngram_split(name,2))
        trigram_len=len(ngram_split(name,3))
        return [word_len,char_len,syl_len,bigram_len,trigram_len]
    except:
        return [0,0,0,0,0]
#计算每个字母出现的频率
def letter_frequency(name):
   frequency = []
   for i in [chr(x) for x in range(97,123)]:
       frequency.append(name.count(i))
   return frequency

#获取属性特征
def cal_tf_idf(queryword,document):
    vectorizer=CountVectorizer()
    X=vectorizer.fit_transform(document)
    df_word = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
    df_word_tf = pd.DataFrame(list(zip(vectorizer.get_feature_names(),df_word.sum()/df_word.values.sum())),columns=['word','tf'])
    transformer = TfidfTransformer(smooth_idf=True,norm='l2',use_idf=True)
    tfidf = transformer.fit_transform(X)
    df_word_idf = pd.DataFrame(list(zip(vectorizer.get_feature_names(),transformer.idf_)),columns=['word','idf'])
    df_word_tf_dict,df_word_idf_dict=dict(),dict()
    for _, row in df_word_tf.iterrows():
        word,tf=row
        df_word_tf_dict.setdefault(word,tf)
    for _, row in df_word_idf.iterrows():
        word,idf=row
        df_word_idf_dict.setdefault(word,idf)
    if queryword in df_word_tf_dict:
        return df_word_tf_dict[queryword],df_word_idf_dict[queryword]
    else:
        return 0.0,0.0
#获取上下文特征
def is_k_th_document(queryword,document,k=5):
    document_len=len(document)
    if k<=document_len:
        if queryword in document[:k]:
            return 1
        else:
            return 0
    else:
        if queryword in document[:document_len]:
            return 1
        else:
            return 0
def co_occurrence_frequency(queryword1,queryword2,document):
    count=0
    for text in document:
        words=text.split()
        if set(queryword1.split()).issubset(set(words)) and set(queryword2.split()).issubset(set(words)):
            count+=1
    co_occur_freq=count/len(document)
    return co_occur_freq
