# -*- coding: utf-8 -*-
"""
@Time ： 2021/5/27 9:51
@Author ： Wanglulu
@File ：your_your_script.py
@IDE ：PyCharm Community Edition
"""

"""
学者画像比赛评测脚本
"""
import json
import argparse

def remove_null(list):
    list = [i for i in list if i != '']
    return list

def compute_other_score(prediction,reference):
    if prediction==reference:
        score=1.0
    else:
        score=0.0
    return score

def compute_Jaccrad(prediction, reference):#reference为参考答案，prediction为预测答案
    prediction=remove_null(prediction)
    reference=remove_null(reference)
    grams_reference = set(reference)#去重；如果不需要就改为list
    grams_prediction=set(prediction)
    temp=0
    for i in grams_reference:
        if i in grams_prediction:
            temp=temp+1
    fenmu=len(grams_prediction)+len(grams_reference)-temp #并集
    if fenmu==0:
        jaccard_coefficient=1.0
    else:
        jaccard_coefficient=float(temp/fenmu)#交集
    return jaccard_coefficient


def home_compute_Jaccrad(prediction, reference):
    prediction=remove_null(prediction)
    reference=remove_null(reference)
    grams_reference = set(reference)#去重；如果不需要就改为list
    grams_prediction=set(prediction)
    temp=0
    for i in grams_reference:
        i=i.strip("https://").strip("http://")
        for j in grams_prediction:
            j=j.strip("https://").strip("http://")
            if j==i:
                temp=temp+1
                break
            elif j.startswith(i):
                temp=temp+1
                break
            elif i.startswith(j) and len(i)-(len(j))<10:
                temp=temp+1
                break
            else:
                pass
    fenmu=len(grams_prediction)+len(grams_reference)-temp #并集
    if fenmu==0:
        jaccard_coefficient=1.0
    else:
        jaccard_coefficient=float(temp/fenmu)#交集
    return jaccard_coefficient


def evaluate(hypothesis_data, reference_data):
    """
    提交文件与训练文件同构；
    """
    scores=[]
    home_scores=[]
    title_scores=[]
    gender_scores=[]
    linenum=0
    for refer in reference_data:
        pred=hypothesis_data[linenum]
        homepage_score=home_compute_Jaccrad(pred["homepage"],refer["homepage"])
        #email_score=compute_Jaccrad(pred["email"],refer["email"])
        title_score=compute_other_score(pred["title"],refer["title"])
        gender_score=compute_other_score(pred["gender"],refer["gender"])
        #language_score=compute_other_score(pred["lang"],refer["lang"])
        each_score=(homepage_score+title_score+gender_score)/3
        scores.append(each_score)
        home_scores.append(homepage_score)
        title_scores.append(title_score)
        gender_scores.append(gender_score)
        linenum+=1
    score_ = sum(scores) / len(scores)
    home_score_=sum(home_scores)/len(home_scores)
    title_score_ = sum(title_scores) / len(title_scores)
    gender_score_=sum(gender_scores)/len(gender_scores)

    return score_, home_score_, title_score_, gender_score_


def main(hypothesis_path, reference_path):
    is_success = False
    score = 0.
    try:
        with open(hypothesis_path,"r",encoding="utf-8") as f:
            hypothesis_data=[]
            for x in f.readlines():
                line=json.loads(x)
                hypothesis_data.append(line)
    except Exception as e:
        err_info = "hypothesis file load failed. please upload a json file such as 'train.json'. err: {}".format(e)
        return score, is_success, err_info

    try:
        with open(reference_path, "r",encoding="utf-8") as f:
            reference_data=[]
            for x in f.readlines():
                line=json.loads(x)
                reference_data.append(line)

    except Exception as e:
        err_info = "reference file load failed. please upload a json file. err: {}".format(e)
        return score, is_success, err_info

    try:
        score_, home_score_, title_score_, gender_score_ = evaluate(hypothesis_data, reference_data)
    except Exception as e:
        return score, is_success, str(e)
    return  score_, home_score_, title_score_, gender_score_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hp",
        type=str,
        # required=True,
        help="hypothesis file",
    )
    parser.add_argument(
        "--rf",
        type=str,
        # required=True,
        help="reference file",
    )
    # parser.add_argument(
    #     "-l",
    #     type=str,
    #     # required=True,
    #     help="result file",
    # )

    args = parser.parse_args()
    # args.hp='qingyou_result.json'
    # args.rf = 'new_ground.json'

    # hypothesis_path = "data/hypothesis.text"  # 学生提交文件
    # reference_path = "data/test1.text"  # 答案文件
    # result_path = "data/result.log"  # 结果文件

    # score,homepage_score,title_score,gender_score = main(args.hp, args.rf)
    # print(score,homepage_score,title_score,gender_score)
    r = main(args.hp, args.rf)
    print(r)
    # with open(args.l, "w", encoding="utf-8") as f:
    #     if success:
    #         f.write("{}###{}".format(score, ret_info))
    #     else:
    #         f.write(ret_info)

