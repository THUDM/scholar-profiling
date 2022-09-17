## bert-baseline算法使用说明
1、安装环境conda
bert4keras、keras、tensorflow

2、文件tools.py提供一些工具函数
（1）函数extract_google_page(file)加载google搜索列表页
（2）函数get_content_pages(id)加载给定id的所有内容页，返回一个列表，列表元素为每个内容页的html文本
（3）函数create_homepage_classification_data()生成“主页分类”训练数据
（4）函数create_title_classification_data()生成“职称分类”训练数据  sed -i -e "s/\r//g" title.train
（5）函数merge_result()合并所有属性

	
3、homepage_classification_bert.py用于训练模型并判断学者的主页，其中train为训练函数，pred为预测函数

4、title_classification_bert.py用于训练模型并判断学者的称职，其中train为训练函数，pred为预测函数

5、get_gender.py用于判断学者的性别，其中pred为预测函数