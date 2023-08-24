import torch
import torch.nn as nn
import numpy as np
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from train_and_eval import MyDataset,Deberta
from sklearn.metrics import roc_auc_score,classification_report
import warnings
warnings.filterwarnings("ignore")

evaluate_path='/home/shishijie/workspace/project/deberta_cls/data/homepage.test'
ptm_path='/home/shishijie/workspace/PTMs/deberta-v3-large'
# ptm_path='/home/shishijie/workspace/PTMs/bert-base-uncased'
# ptm_path='/home/shishijie/workspace/PTMs/xlm-roberta-base'
# ptm_path='/home/shishijie/workspace/PTMs/albert-base-v2'
# ptm_path='/home/shishijie/workspace/PTMs/deberta-base'

max_length=128
batch_size=32
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_dataset=MyDataset(evaluate_path,ptm_path,max_length)
test_loader=DataLoader(test_dataset,batch_size,shuffle=False)


def test(model_path):
    model=torch.load(model_path)
    model.to(device)
    model.eval()
    softmax=nn.Softmax(dim=-1)

    total_length=len(test_loader.dataset)
    correct=0
    total_label=np.array([])
    total_pred=np.array([])
    for input,label in tqdm(test_loader,desc='testing'):
        
        for k,v in input.items():
            input[k]=v.to(device)
        
        output=model(input)
        output=softmax(output)
        # print(output)
        output=torch.argmax(output,dim=-1)
        output=output.cpu().numpy()
        # print(output)
        label=np.array(label)
        
        correct+=sum(output==label)
        total_pred=np.concatenate((total_pred,output))
        total_label=np.concatenate((total_label,label))
    acc=correct/total_length
    auc=0
    if 'gender' in model_path:
        auc=roc_auc_score(total_label,total_pred)
    report=classification_report(total_label,total_pred)
    metrics={'acc':acc,'auc':auc,'report':report,'pred':total_pred.tolist()}
    print('acc:',acc,'auc',auc)
    
    return metrics


def remove_null(list):
    list = [i for i in list if i != '']
    return list

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

def home_evaluate(pred,home_test):
    global_name=''
    pred_test=[]
    global_index=-1
    for idx,item in enumerate(pred):
        text=home_test[idx]
        name=text.split('[SEP]')[0]
        url=text.split('[SEP]')[1].split('\t')[0]
        
        if global_name!=name:
            pred_test.append([])
            global_name=name
            global_index+=1
            
        if item==1:
            pred_test[global_index].append(url)
    
    ground_truth=[]
    with open('/home/shishijie/workspace/project/scholar/data/ground_truth.json','r',encoding='utf8') as fr:
        for item in fr.readlines():
            item=json.loads(item)
            ground_truth.append('homepage')
                
    homepage_score=[]
    for x,y in zip(pred_test,ground_truth):
        score=home_compute_Jaccrad(x,y)
        homepage_score.append(score)
    
    home_score=sum(homepage_score)/len(homepage_score)
    return home_score
        

def main(path):
    model_name_list=os.listdir(path)
    
    best_epoch=0
    best_acc=0
    auc=0
    
    best_auc=0
    best_auc_epoch=0
    acc=0
    report=[]
    homepage_score=0
    best_home_epoch=0
    
    if 'homepage' in path:
        home_test=[]
        with open('./data/homepage.test','r',encoding='utf8') as fr:
            for line in fr.readlines():
                home_test.append(line.strip())
    
    for idx,model_name in enumerate(model_name_list):
        model_path=os.path.join(path,model_name)
        print(f'\n第 {idx+1} 轮测试:')
        metrics=test(model_path)
        
        report.append(metrics['report'])
        
        if metrics['acc']>best_acc:
            best_acc=metrics['acc']
            best_epoch=idx+1
            auc=metrics['auc']
        
        if metrics['auc']>best_auc:
            best_auc=metrics['auc']
            best_auc_epoch=idx+1
            acc=metrics['acc']
        
        if 'homepage' in path:
            home_score=home_evaluate(metrics['pred'],home_test)
            print(f'home_score:{home_score}')
            if home_score>homepage_score:
                homepage_score=home_score
                best_home_epoch=idx+1
    if 'homepage' in path:
        print(f'\n最优home_score:{homepage_score}\t最优轮次:{best_home_epoch}')
    else:
        print(f'\n最优acc:{best_acc}\t最优轮次:{best_epoch}\tauc:{auc}')
        print(f'\n最优auc:{best_auc}\t最优轮次:{best_auc_epoch}\tacc:{acc}')
        print('\n',report[best_epoch-1])

if __name__=='__main__':
    model_path='/home/shishijie/workspace/project/deberta_cls/model_save/homepage/deberta-v3/'
    main(model_path)