#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import numpy as np
from utils import create_batch_file
from utils import show_progress

import json
import pickle
import shutil
import os 
import glob

import re 

from pprint import pprint
import time


# In[ ]:


##需要重写的方法：
#    build_vocabulary();build_models();build_pipelines();
#    build_optimizer(params);build_scheduler(optimizer);build_batch(batch_id);
#    test_worker();app_worker();
class End2EndBase(object):
    ##args(在代码中出现):
    #scheduler:step_size,step_decay,lr_schedule('warm_up' or 'build_in'),warmup_step
    #model:base_model_dir(path),train_model_dir(path),best_model(编号，str),model_size
    #train:train_base_model(Bool),continue_training(Bool)
    #      n_epoch,data_dir(path),task,file_name,is_lower(Bool)
    #      grad_clip,checkpoint,learning_rate
    def __init__(self,args):        
        self.args=args
        self.base_models={}
        self.train_models={}
        self.batch_data={}            #用来记录当前的数据
        self.test_data={}
        self.global_steps=0           #用来记录训练的总步数
        self.report = []              #记录训练过程中的损失、学习率等
        self.report_dict = {}         #用来记录单次 report

        
    ##需要重写
    ##构建字典的方法
    def build_vocabulary(self):        
        raise NotImplementedError
        
        
    ##设置手动调整学习率的方法，使用StepLR
    def build_scheduler(self,optimizer):        
        scheduler=torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.args.step_size,
            gamma=self.args.step_decay
        )
        return scheduler
    
    
    ##需要重写
    ##构建模型字典
    #base_models:{'name1':model1,'name2':model2}
    #train_models:{'name1':model1,'name2':model2}
    def build_models(self):        
        raise NotImplementedError
        
        
    ##导入base model的参数
    #model（层）参数的储存文件格式：
    #  base:{base_model_dir}/{model_name}.model
    #  train:{train_model_dir}/{model_name}_{epoch_id}_{batch_id}.model
    def init_base_model_params(self):
        for model_name in self.base_models:
            model_path=os.path.join(self.args.base_model_dir,model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(model_path, map_location=lambda storage, loc:storage)
            )
            
            
    ##导入train model的参数
    # 基本不用
    def init_train_model_params(self):
        for model_name in self.train_models:
            model_path=os.path.join(
                self.args.train_model_dir,
                model_name+'_'+str(self.args.best_model)+'.model'
            )
            self.train_models[model_name].load_state_dict(
                torch.load(model_path,map_location=lambda storage, loc:storage)
            )
            
        
    ##需要重写
    ##搭建模型(前向传播和损失函数)
    def build_pipelines(self):
        raise NotImplementedError
        
    
    ##需要重写
    ##定义优化器
    def build_optimizer(self,params):
        raise NotImplementedError
        
        
    ##需要重写
    ##定义在文件中读取batch数据的方法
    #把batch_id转化为batch数据，存入self.batch_data
    def build_batch(self,batch_id):
        raise NotImplementedError
        
        
    ##需要重写    
    #用于模型测试时的前向传播
    def test_worker(self):
        raise NotImplementedError
        
        
    ##需要重写    
    def app_worker(self):
        raise NotImplementedError

    def write_report(self, epoch):
        if not os.path.exists('../nats_results'):
            os.mkdir('../nats_results')
        fout = open(os.path.join('../nats_results', 'report_{}.txt'.format(epoch)), 'w')
        if len(self.report) > 0:
            for itm in self.report:
                json.dump(itm, fout)
                fout.write('\n')
        fout.close()

    ##训练
    def train(self):
        #模型的定义及参数统计
        self.build_vocabulary()
        self.build_models()
        pprint(self.base_models)
        pprint(self.train_models)
        
        self.init_base_model_params()
        params=[]
        for model_name in self.train_models:
            params+=list(self.train_models[model_name].parameters())
        if self.args.train_base_model:
            for model_name in self.base_models:
                params+=list(self.base_models[model_name].parameters())
        p_num=sum([para.numel() for para in params])
        print('Total number of trainable parameters {}.'.format(p_num))
        
        #模型训练的准备工作
        optimizer=self.build_optimizer(params)
        if self.args.lr_schedule=='build-in':
            scheduler=self.build_scheduler(optimizer)
            
        out_dir=os.path.join('..','nats_results')     #模型存储目录
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        uf_model=[0,-1]
        if self.args.continue_training:               #如果继续训练的话，挑选目录下训练最久的模型为训练对象
            model_para_path=glob.glob(os.path.join(out_dir,'*.model'))
            
            if len(model_para_path)>0:
                uf_model=[]
                for path in model_para_path:
                    arr=re.split('/', path)[-1]
                    arr=re.split('_|\.', arr)
                    arr=[int(arr[-3]),int(arr[-2])]
                    if not arr in uf_model:
                        uf_model.append(arr)
                cc_model=sorted(uf_model)[-1]              #cc_model是训练时间最久的模型的编号
            
                #导入模型(某个模型导入失败时，导入另外一个)
                try:
                    print('Try *_{}_{}.model'.format(cc_model[0],cc_model[1]))
                    for model_name in self.train_models:
                        m_path=os.path.join(
                            out_dir, model_name+'_'+str(cc_model[0])+'_'+str(cc_model[1])+'.model')
                        self.train_models[model_name].load_state_dict(
                            torch.load(m_path, map_location=lambda storage, loc:storage))
                except:
                    cc_model=sorted(uf_model)[-2]
                    print('Try *_{}_{}.model'.format(cc_model[0],cc_model[1]))
                    for model_name in self.train_models:
                        m_path=os.path.join(
                            out_dir, model_name+'_'+str(cc_model[0])+'_'+str(cc_model[1])+'.model')
                        self.train_models[model_name].load_state_dict(
                            torch.load(m_path, map_location=lambda storage, loc:storage))
                print('Continue training with *_{}_{} model'.format(cc_model[0],cc_model[1]))
                uf_model=cc_model
            
        else:                                          #不是继续训练的状态，则要清空目录
            shutil.rmtree(out_dir) 
            os.mkdir(out_dir)
            
        f=open(os.path.join(out_dir,'args.pickled'),'wb')
        pickle.dump(self.args, f)
        f.close()
        
        start_time=time.time()
        cclb=0
        
        #开始训练
        n_epoch=self.args.n_epoch
        for epoch in range(uf_model[0],n_epoch):
            #create_batch_file用来构造需要遍历的数据文件(每个epoch都可以构造)，它会返回batch总数
            #这种方法节省了内存，但是更加耗时
            #参数有（path_data, path_work, is_shuffle, fkey_, file_, batch_size, is_lower=True）
            #数据来源地址：{path_data}/{file_}
            #数据输出地址：{path_work}/batch_{fkey_}_{batch_size}
            n_batch=create_batch_file(
                path_data=self.args.data_dir,
                path_work=os.path.join('..','nats_results'),
                is_shuffle=True,
                fkey_=self.args.task,
                file_=self.args.file_train,
                batch_size=self.args.batch_size
            )
            print('The number of batch is {}.'.format(n_batch))
            
            self.global_steps=max(0, epoch)*n_batch
            for batch_id in range(n_batch):
                self.global_steps+=1
                
                #学习率调度策略为‘warm-up’时，更新学习率
                learning_rate = self.args.learning_rate
                if self.args.lr_schedule=='warm-up':          #global_steps=warmup_steps时有最大值，然后随着global_steps的增大而减小
                    learning_rate=2.0*(
                        self.args.model_size ** (-0.5) *
                         min(self.global_steps ** (-0.5),
                             self.global_steps * self.args.warmup_step ** (-1.5)))
                    for group in optimizer.param_groups:
                        group['lr']=learning_rate
                        
                #学习率调度策略为‘build-in’时，读取学习率
                elif self.args.lr_schedule=='build-in':
                    for group in optimizer.param_groups:
                        learning_rate=group['lr']
                        break
                        
                #保证从batch_id的地方开始
                if cclb == 0 and batch_id < n_batch-1 and batch_id <= uf_model[1]:
                    continue
                else:
                    cclb += 1
                    
                self.build_batch(batch_id)
                loss=self.build_pipelines()
                
                if loss != loss:
                    raise ValueError
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip)
                optimizer.step()

                if 'lr' in self.args.report:
                    self.report_dict['lr'] = np.around(learning_rate, 6)
                    self.report_dict['step'] = cclb
                if 'loss' in self.args.report:
                    self.report_dict['loss'] = np.around(loss.data.cpu().numpy(), 4).tolist()
                    self.report_dict['step'] = cclb

                self.report.append(self.report_dict)
                self.report_dict = {}
                
                #打印信息
                if batch_id % 5 == 0:
                    end_time=time.time()
                    print('epoch:{}, batch:{}/{}, lr:{}, loss:{}, time:{}h'.
                         format(epoch,batch_id,n_batch,
                                np.around(learning_rate,6),
                                np.round(float(loss.data.cpu().numpy()),4),
                                np.round((end_time-start_time)/3600.0,4)),
                          end='\r'
                        )
                del loss
                #保存模型    
                if batch_id % self.args.checkpoint == 0:
                    for model_name in self.train_models:
                        if epoch >= 0.8*n_epoch \
                                or epoch == n_epoch-1 \
                                or epoch == n_epoch-2:
                            out_path=os.path.join(
                                out_dir, model_name+'_{}_{}.model'.format(str(epoch),str(batch_id)))
                        
                        else:
                            out_path=os.path.join(
                                out_dir, model_name+'_{}_0.model'.format(str(epoch)))
                        fmodel=open(out_path,'wb')
                        torch.save(self.train_models[model_name].state_dict(), fmodel)
                        fmodel.close()
                    
            #每个epoch结束保存模型
            for model_name in self.train_models:
                if epoch >= 0.8*n_epoch or \
                        epoch == n_epoch-1 or \
                        epoch == n_epoch-2:
                    out_path=os.path.join(
                            out_dir, model_name+'_{}_{}.model'.format(str(epoch),str(batch_id)))
                        
                else:
                    out_path=os.path.join(out_dir,
                                        model_name+'_{}_0.model'.format(str(epoch)))
                fmodel=open(out_path,'wb')
                torch.save(self.train_models[model_name].state_dict(), fmodel)
                fmodel.close()  
            print('')

            if self.args.lr_schedule == 'build-in':
                scheduler.step()

            self.write_report(epoch)
            self.report = []







