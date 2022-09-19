# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/26 11:25
@Author ： Wanglulu
@File ：record_classify.py
@IDE ：PyCharm Community Edition
"""
import re
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json
from generate_features import *
from get_abstract import *
import pickle
from collections import Counter
from pypinyin import lazy_pinyin


def ischname2en(name):
    if re.compile(u'[\u4e00-\u9fa5]').search(name):
        name_list=lazy_pinyin(name)
        familyname = name_list[0]
        given_name_list = name_list[1:]
        given_name = ""
        for y in given_name_list:
            given_name = given_name + y
        en_name = given_name + " " + familyname
        en_name = en_name.title().lstrip()
        return en_name
    else:
        return name


def feature(pandas_file):
    features=[]
    for i,row in pandas_file.iterrows():
        if i % 100 == 0:
            print("feature", i)
        id=row["id"]
        name=row["name"]
        org=row["org"]
        name=ischname2en(name)
        name=unicodeToAscii(name)
        #print (id,name)
        #gender=row["gender"]
        urls,item,abstract=get_info(id)
        length_feature=get_length_feature(name)
        letter_frequency_feature=letter_frequency(name)
        try:
            corpus=[s.lower() for s in abstract]
            tf_m,idf_m=cal_tf_idf("his",corpus)
            tf_f,idf_f=cal_tf_idf("her",corpus)
            m_isink=is_k_th_document("his",corpus,k=10)
            f_isink=is_k_th_document("her",corpus,k=10)
            m_org_item_co=co_occurrence_frequency("his",org,item)
            m_org_abstract_co=co_occurrence_frequency("his",org,abstract)
            f_org_item_co=co_occurrence_frequency("her",org,item)
            f_org_abstract_co=co_occurrence_frequency("her",org,abstract)
            first_name=name.split()[0]
            last_name=name.split()[-1]
            m_first_item_co=co_occurrence_frequency("his",first_name,item)
            m_first_abstract_co=co_occurrence_frequency("his",first_name,abstract)

            m_last_item_co=co_occurrence_frequency("his",last_name,item)
            m_last_abstract_co=co_occurrence_frequency("his",last_name,abstract)

            f_first_item_co=co_occurrence_frequency("her",first_name,item)
            f_first_abstract_co=co_occurrence_frequency("her",first_name,abstract)

            f_last_item_co=co_occurrence_frequency("her",last_name,item)
            f_last_abstract_co=co_occurrence_frequency("her",last_name,abstract)
            feature=[id,name,tf_m,idf_m,tf_f,idf_f,m_isink,f_isink,m_org_item_co,m_org_abstract_co,f_org_item_co,f_org_abstract_co,m_first_item_co,m_first_abstract_co,m_last_item_co,m_last_abstract_co,f_first_item_co,f_first_abstract_co,f_last_item_co,f_last_abstract_co]
            #feature.extend(length_feature)
            # feature.extend(letter_frequency_feature)
            features.append(feature)

        except:
            #feature=[name,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,df['gender'][index]]
            feature=[id,name,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            #feature.extend(length_feature)
            # feature.extend(letter_frequency_feature)
            features.append(feature)
    return features

# df=pd.read_excel("/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_train.xlsx",keep_default_na=False)
# df=pd.read_excel("../../data/new_train.xlsx",keep_default_na=False)
df=pd.read_excel(os.path.join(settings.DATA_DIR, "raw", "new_train.xlsx"), keep_default_na=False)
# df2=pd.read_excel("train2.xlsx",keep_default_na=False)
#
# df=df1.append(df2)
df["gender"].replace({"female":1,"male":0,"no_records":2,"":2,"unknown":2},inplace=True)
#print (df["gender"])
features=feature(df)
print("特征转换完成")
#encoder=preprocessing.OneHotEncoder(handle_unknown="ignore")

x_train= [feature[2:-1] for feature in features]
# print (x_train_data)
# encoder.fit(x_train_data)
# x_train=encoder.transform(x_train_data).toarray()
# x_train=preprocessing.scale(x_train)
y_train=df['gender'].values.tolist()

#决策树
df_clf = DecisionTreeClassifier()
dt_model = df_clf.fit(x_train,y_train)
os.makedirs("save/", exist_ok=True)
with open('save/dt.pickle', 'wb') as f:
    pickle.dump(dt_model, f)
#SVM
svm_clf = Pipeline(( ("scaler", StandardScaler()),
                     ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))
svm_model=svm_clf.fit(x_train,y_train )
with open('save/svm.pickle', 'wb') as f:
    pickle.dump(svm_model, f)

#lR
lr_clf = LogisticRegression(penalty="l2")
lr_model = lr_clf.fit(x_train,y_train)
with open('save/lr.pickle', 'wb') as f:
    pickle.dump(lr_model, f)
#随机森林
rf_clf = RandomForestClassifier()
rf_model=rf_clf.fit(x_train,y_train)
with open('save/rf.pickle', 'wb') as f:
    pickle.dump(rf_model, f)


knn = KNeighborsClassifier()
knn_model=knn.fit(x_train,y_train)
with open('save/knn.pickle', 'wb') as f:
    pickle.dump(knn_model, f)

# df_test=pd.read_excel("/DATA/disk1/model_data/wll_data/kaiyu/ccks_numberone/CCKS2021_Aminer_profiling_googlesearch/dataset/new_test.xlsx",keep_default_na=False)
# df_test=pd.read_excel("../../data/new_test.xlsx",keep_default_na=False)
df_test = pd.read_excel(os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx"), keep_default_na=False)
test_features=feature(df_test)
x_test=[feature[2:-1] for feature in test_features]


f1=open('save/dt.pickle', 'rb')
clf1 = pickle.load(f1)

f2=open('save/lr.pickle', 'rb')
clf2 = pickle.load(f2)

f3=open('save/rf.pickle', 'rb')
clf3 = pickle.load(f3)

f4=open('save/svm.pickle', 'rb')
clf4 = pickle.load(f4)
f5=open('save/knn.pickle', 'rb')
clf5 = pickle.load(f5)

y_pred_1=clf1.predict(x_test)
y_pred_2=clf2.predict(x_test)
y_pred_3=clf3.predict(x_test)
y_pred_4=clf4.predict(x_test)
y_pred_5=clf5.predict(x_test)
y_pred=[]
id=0
for i in y_pred_1:
    pred_list=[i,y_pred_2[id],y_pred_2[id],y_pred_4[id],y_pred_5[id]]
    d = Counter(pred_list)
    pred=d.most_common(1)[0][0]
    y_pred.append(pred)
    id+=1

df_test["gender_record"]=y_pred
df_test["gender_record"].replace({1:"female",0:"male",2:""},inplace=True)
os.makedirs("output/sml/", exist_ok=True)
fw=open("output/sml/gender_predict.json",'w',encoding="utf-8")
for i,row in df_test.iterrows():
    id=row["id"]
    name=row["name"]
    org=row["org"]
    gender=row["gender_record"]
    predict_data={"id":id,"name":name,"org":org,"gender":gender}
    title_data=json.dumps(predict_data)
    fw.write(title_data+"\n")
fw.close()
