import torch
import random
import time
import numpy as np
from transformers import AutoTokenizer,AutoModel,AutoConfig,AutoModelForSequenceClassification
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch import relu
from sklearn.metrics import classification_report

class CFG:
    pretrain_model_path='/home/shishijie/workspace/PTMs/zyBert/tcmbert13500'
    
    train_data_path='/home/shishijie/workspace/project/deberta_cls/data/train.txt'
    test_data_path='/home/shishijie/workspace/project/deberta_cls/data/test.txt'
    device=torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    
    seed=25
    
    # cnn
    num_filters=128
    filter_sizes=[3,4,5]
    is_cnn=True
    
    warmup_rate=0.1
    batch_size=32
    max_length=256
    classes=13
    lr=1e-5
    epochs=20
    
    log_steps=10

# 设置种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)#numpy产生的随机数一致
    random.seed(seed)
    # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    torch.backends.cudnn.deterministic = True
    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    torch.backends.cudnn.benchmark = False

set_seed(CFG.seed)

# 数据处理
class MyDataset(Dataset):
    def __init__(self,data_path,ptm_path,max_length=512) -> None:
        super().__init__()
        self.tokenizer=AutoTokenizer.from_pretrained(ptm_path)
        
        self.data,self.label=self.get_data(data_path)
        self.label=torch.LongTensor(self.label)
        self.length=len(self.data)
        self.tokenized_data=self.tokenize_data(self.tokenizer,self.data,max_length)

    def get_data(self,path):
        data=[]
        label=[]
        with open(path,'r',encoding='utf8') as fr:
            for line in fr.readlines():
                line=line.strip().split('\t')
                data.append(line[0])
                label.append(int(line[1]))
        return data,label
    
    def tokenize_data(self,tokenizer,data,max_length):
        tokenizer_data=[]
        for item in tqdm(data,desc='Tokenizing'):
            tokenizer_data.append(
                tokenizer(item,truncation=True,padding='max_length',max_length=max_length,return_tensors='pt')
            )
        return tokenizer_data
    def __len__(self):
        return self.length
    
    def __getitem__(self, index) :
        input=self.tokenized_data[index]
        label=self.label[index]
        
        for k,v in input.items():
            input[k]=v.squeeze(0)
        
        
        return input,label

# 模型定义
class MyBert(nn.Module):
    def __init__(self,ptm_path,classes,num_filters=None,filter_sizes=None,is_cnn=False) -> None:
        super().__init__()
        
        self.bert=AutoModel.from_pretrained(ptm_path)
        self.config=AutoConfig.from_pretrained(ptm_path)
        
        self.is_cnn=is_cnn
        if self.is_cnn:     
            self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, self.bert.config.hidden_size)) for fs in filter_sizes])
        self.dropout=nn.Dropout(self.config.hidden_dropout_prob)
        
        if self.is_cnn:
            self.linear=nn.Linear(num_filters * len(filter_sizes), classes)
        else:
            self.linear=nn.Linear(self.config.hidden_size,classes)        
    
    def conv_and_pool(self, x, conv):
        # 这里加了一层relu
        # x:[batch_size,max_length,hidden_size]
        x = F.relu(conv(x)).squeeze(3)  # x:[batch_size,num_filters,max_length-filter_size+1,1].squeeze(-1)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)   # x:[batch_size,num_filters,1].squeeze(-1)
        return x
    
    def forward(self,input):
        
        outputs=self.bert(**input,output_hidden_states=True)
        
        last_hidden_state=outputs.last_hidden_state # [batch_size,seq_len,hidden]
        
        if self.is_cnn:        
            x = last_hidden_state.unsqueeze(1)  # 为卷积操作增加维度
            x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
            # x = [relu(conv(x)).squeeze(3) for conv in self.convs]  # 卷积操作和激活函数
            # x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]  # 最大池化
            # x = torch.cat(x, 1)  # 拼接不同尺寸卷积的池化结果
            x = self.dropout(x)
            logits = self.linear(x)  # 全连接层
        else:
            
            # hidden_state=outputs.hidden_states
            # first=hidden_state[1]
            # last=hidden_state[-1]
            # first_avg=torch.mean(first,dim=1,keepdim=True)
            # last_avg=torch.mean(last,dim=1,keepdim=True)
            # avg=torch.cat((first_avg,last_avg),dim=1)
            # output=torch.mean(avg,dim=1)
            
            output=last_hidden_state[:,0,:] # 取CLStoken 的向量
            # output=torch.mean(last_hidden_state,dim=1) # [batch_size,hidden]
            output = self.dropout(output)
            logits=self.linear(output)                 # [bath_size,classes]
        
        return logits

def eval(model,test_loader):
    model.eval()
    correct=0
    softmax=nn.Softmax(dim=-1)
    total_label=np.array([])
    total_pred=np.array([])
    with torch.no_grad():
        for input,label in tqdm(test_loader,desc='testing'):
            
            for k,v in input.items():
                input[k]=v.to(CFG.device)
                
            output=model(input)
            output=softmax(output)
            output=torch.argmax(output,dim=-1)
            output=output.cpu().numpy()
            label=np.array(label)
               
            correct+=sum(output==label)
            total_pred=np.concatenate((total_pred,output))
            total_label=np.concatenate((total_label,label))
    report=classification_report(total_label,total_pred)
    acc=correct/len(test_loader.dataset)
    print(f'acc:{acc}')
    model.train()
    mertrics={'acc':acc,'report':report}
    return mertrics

def main():
    train_dataset=MyDataset(CFG.train_data_path,CFG.pretrain_model_path,CFG.max_length)
    test_dataset=MyDataset(CFG.test_data_path,CFG.pretrain_model_path,CFG.max_length)
    
    train_loader=DataLoader(train_dataset,batch_size=CFG.batch_size,shuffle=False)
    test_loader=DataLoader(test_dataset,batch_size=CFG.batch_size,shuffle=False)
    
    model=MyBert(CFG.pretrain_model_path,CFG.classes,CFG.num_filters,CFG.filter_sizes,is_cnn=CFG.is_cnn)

    criterion=nn.CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_group_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer=AdamW(optimizer_group_parameters,lr=CFG.lr)

    num_training_steps=len(train_loader)*CFG.epochs
    num_warmup_steps=int(num_training_steps*CFG.warmup_rate)
    scheduler=get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        num_cycles=0.5
        )

    model.to(CFG.device)
    global_steps=0
    training_loss=0
    best_val_acc=0
    report=None
    for epoch in range(1,CFG.epochs+1):
        for batch_idx,(input,label) in enumerate(train_loader):
            
            global_steps+=1
            
            label=label.to(CFG.device)
            for k,v in input.items():
                input[k]=v.to(CFG.device)
                
            output=model(input)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            training_loss+=loss.item()
            # if global_steps%CFG.wandb_log_steps==0:
            #     wandb.log({'loss':loss.item(),
            #                'lr':optimizer.param_groups[0]['lr'],
            #                'global_loss':training_loss/global_steps,
            #                'epoch':epoch},
            #                 step=global_steps)   
            
            if global_steps%CFG.log_steps==0:
                
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(input['input_ids']),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
        # test
        mertrics=eval(model,test_loader)
        # wandb.log({'val_acc':acc},
        #           step=global_steps)
        if mertrics['acc']>best_val_acc:
            best_val_acc=mertrics['acc']
            best_epoch=epoch
            report=mertrics['report']
            # print(f'best_val_acc:{best_val_acc}\t模型保存在:{save_path}')
            # torch.save(model,save_path+f'model.pth')
    print(f'best_val_acc:{best_val_acc}\t epoch:{best_epoch}\n{report}')

if __name__=='__main__':
    main()