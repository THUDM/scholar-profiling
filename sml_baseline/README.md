## xgboost-baseline算法使用说明
1、安装环境conda
xgboost 1.5.2、sklearn、pickle、jieba 0.42.1

2、职称抽取：TitlePrediction文件夹
	程序入口为title_main.py，读取id、对应的所有html页面。
	设定职称对应表，用正则表达式匹配抽取，条件是提取目前的职称（过滤过去式的句子），最后根据所有网页的职称次数出现频次最高的职称作为最终结果
	
	
3、性别预测：GenderPrediction文件夹
	程序入口为main.py，读取id、对应的所有html页面。设计特征，主要是name、org、his 或her等在搜索记录出现的特征，作为属性特征，用多个分类器进行投票。


4、主页预测：HomepagePrediction文件夹
   训练程序入口为homepage_train.py
   程序入口为clean_homepage.py, 读取人名机构、对应的html s1 s2检索页面。
   设计人名机构和检索到的url的特征，将检索结果中的摘要转换为TFIDF向量，
   将两个特征向量进行拼接，使用XGBoost对拼接后向量进行分类。