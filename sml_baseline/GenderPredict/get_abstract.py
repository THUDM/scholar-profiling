# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 16:20
@Author ： Wanglulu
@File ：get_abstract.py
@IDE ：PyCharm Community Edition
"""
import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

import settings


def parse(html):
    urls=[]
    titles=[]
    contents=[]
    soup = BeautifulSoup(html,'lxml')
    URLs_src = soup.find_all('div',{'class', 'ZINbbc xpd O9g5cc uUPGi'})

    for val in URLs_src:
        k=val.find('a')['href']
        title_=val.find('h3')
        content_=val.find('div',{'class', 'BNeawe s3v9rd AP7Wnd'})

        try:
            m = re.search("(?P<url>https?://[^\s]+)", k)
            n = m.group(0)
            url = n.split('&')[0]
            domain = urlparse(url)
            if(re.search('google.com', domain.netloc)):
                continue
            else:
                urls.append(url)
                titles.append(title_.getText())
                contents.append(content_.getText())
        except:
            continue

    return urls,titles,contents

def get_info(id):
    # html_path="/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/data/"
    # html_path="/home/zfj/research-data/user_profiling/googleSearch/"
    html_path = os.path.join(settings.DATA_DIR, "googleSearch", "data")
    urls=[]
    titles=[]
    contents=[]
    try:
        # with open(html_path+id+"_s1.html","r",encoding="utf-8") as f1:
        with open(os.path.join(html_path, id+"_s1.html"), "r", encoding="utf-8") as f1:
            html_1=f1.read()
            urls_1,titles_1,contents_1=parse(html_1)
            urls.extend(urls_1)
            titles.extend(titles_1)
            contents.extend(contents_1)
    except:
        pass
    try:
        # with open(html_path+id+"_s2.html","r",encoding="utf-8") as f2:
        with open(os.path.join(html_path, id+"_s2.html"), "r", encoding="utf-8") as f2:
            html_2=f2.read()
            urls_2,titles_2,contents_2=parse(html_2)
            urls.extend(urls_2)
            titles.extend(titles_2)
            contents.extend(contents_2)
    except:
        pass
    return urls,titles,contents

# df1=pd.read_excel("dev_for_test.xlsx")
#
# df1['urls']= df1["id"].apply(lambda x : get_info(x)[0])
# df1['titles']= df1["id"].apply(lambda x : get_info(x)[1])
# df1['contents']= df1["id"].apply(lambda x : get_info(x)[2])
# df1.to_excel("dev_for_test_utc.xlsx")

# df2=pd.read_excel("train2.xlsx")

# df2['urls']= df2["id"].apply(lambda x :zidian  get_info(x)[0])
# df2['titles']= df2["id"].apply(lambda x : get_info(x)[1])
# df2['contents']= df2["id"].apply(lambda x : get_info(x)[2])
#
# df2.to_excel("train2_utc.xlsx")



