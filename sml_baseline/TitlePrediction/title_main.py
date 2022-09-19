# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 16:20
@Author ： Wanglulu
@File ：get_abstract.py
@IDE ：PyCharm Community Edition
"""
import random
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import json
from urllib.parse import urlparse
import html2text
# import trafilatura
import title_predict
from collections import Counter
from lxml import etree

import settings


def get_row_text(html):
    try:
        text=html2text.html2text(html)
        text="\n".join([s.replace("\n", "") for s in text.split("\n\n") if s.strip()])
        return text
    except Exception as e:
        pass

#获取每条记录的url、标题和摘要
def parse_search(html):
    urls=[]
    titles=[]
    abtracts=[]
    root = etree.HTML(html)
    url_title_src=root.xpath('.//div[@class="ZINbbc xpd O9g5cc uUPGi"]')
    for val_ in url_title_src:
        urls_src=val_.xpath('.//div[@class="kCrYT"]/a[1]/@href')
        for val_url in urls_src:
            try:
                m = re.search("(?P<url>https?://[^\s]+)", val_url)
                n = m.group(0)
                url = n.split('&')[0]
                urls.append(url)
                titles.append(val_.xpath('string(..//div[@class="kCrYT"]//a[1]//div[@class="BNeawe vvjwJb AP7Wnd"])'))
                abtracts.append(val_.xpath('string(.//div[@class="BNeawe s3v9rd AP7Wnd"]//div[@class="BNeawe s3v9rd AP7Wnd"])'))
            except:
                continue
    return urls,titles,abtracts

def get_info(html_path,id):
    urls=[]
    titles=[]
    contents=[]
    try:
        # with open(html_path+id+"_s1.html","r",encoding="utf-8") as f1:
        with open(os.path.join(html_path, id+"_s1.html"), "r", encoding="utf-8") as f1:
            html_1=f1.read()
            urls_1,titles_1,contents_1=parse_search(html_1)
            urls.extend(urls_1)
            titles.extend(titles_1)
            contents.extend(contents_1)
    except:
        pass
    return urls,titles,contents

# df=pd.read_excel("/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_test.xlsx",keep_default_na=False)
# df=pd.read_excel("../../data/new_test.xlsx",keep_default_na=False)
df = pd.read_excel(os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx"), keep_default_na=False)


#搜索引擎地址
# html_path="/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/data/"
# html_path="/home/zfj/research-data/user_profiling/googleSearch/"
html_path = os.path.join(settings.DATA_DIR, "googleSearch", "data")


#无效噪声网页
neg_urls=["www.researchgate.net","pubmed.ncbi.nlm.nih.gov"]
pattern="|".join(neg_urls)

position_guesser = title_predict.title_predict()
result=[]
for i,row in df.iterrows():
    list_title=[]
    id=row["id"]
    name=row["name"]
    org=row["org"]
    position_truth=row["title"]
    urls,titles,abstracts=get_info(html_path,id)
    positions=[]
    url_id=0
    for title in titles:
        try:
            all_text=""
            abstract=abstracts[url_id]
            url=urls[url_id]
            domain = urlparse(url)
            if(re.search(pattern, domain.netloc)):
                ta=title+abstract
                position=position_guesser.pos_guess_ta(name.lower(),ta.lower())
            else:
                ta=title+abstract
                position=position_guesser.pos_guess_ta(name.lower(),ta.lower())
                if os.path.isfile(html_path+id+"_"+str(url_id)+".html"):
                    with open(html_path+id+"_"+str(url_id)+".html","r",encoding="utf-8") as fr_subhtml:
                        subhtml=fr_subhtml.read()
                        content=get_row_text(subhtml)
                        #print (content)
                        position_1=position_guesser.pos_guess_ta(name.lower(),ta.lower())
                        position_2=position_guesser.pos_guess_html(name.lower(),content.lower())
                        position=position_1+position_2
            positions.extend(position)
        except Exception as e:
            pass

        url_id+=1

    if positions:#取出现频次最多的职称
        d = Counter(positions)
        position_=d.most_common(1)[0][0]
    else:
        position_="Other(其他)"

    print (id,position_,position_truth)
    title_data={"id":id,"name":name,"org":org,"title":position_}
    result.append(title_data)
os.makedirs("output/sml/", exist_ok=True)
with open("output/sml/test_title_predict1.json",'w',encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=4)
