#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

import glob
import json
import os

from End2EndBase import End2EndBase
from utils import create_batch_file, show_progress

import time
from pprint import pprint

import re


# In[ ]:


##需要重写的方法：(新增了两个)
#    evaluate_worker(input_);aspect_worker(self);
#    build_vocabulary();build_models();build_pipelines();
#    build_optimizer(params);build_batch(batch_id);
#    test_worker();app_worker();
##重写了：
#    validate();test();
class End2EndAspDecBase(End2EndBase):
    def __init__(self, args=None):
        ##args(在代码中出现):
        #    验证: optimal_model_dir; data_dir;task; file_dev; batch_size
        #    测试: data_dir; task; file_test; batch_size;
        #          use_optimal_model; optimal_model_dir; model_optimal_key; file_out
        super().__init__(args=args)
        self.test_data=[]               #用来记录验证\测试结果
        
        
    ##需要重写
    ##用来评价模型，会返回f1
    def evaluate_worker(self,input_):
        raise NotImplementedError

    
    ##验证各模型，并保存最有优模型的参数
    def validate(self):
        self.build_vocabulary()
        self.build_models()
        pprint(self.base_models)
        pprint(self.train_models)
        self.init_base_model_params()
        
        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
            
        with torch.no_grad():
            
            model_para_files=glob.glob(os.path.join(
                '../nats_results', sorted(list(self.train_models))[0]+'*.model'))
            #在训练目录下找到模型的 epoch 和 batch_id ,并排序，
            #输出到 model_para_list:list([epoch, batch_id, file_name])
            for i in range(len(model_para_files)):
                arr=re.split('\.|_',model_para_files[i])
                arr=[int(arr[-3]),int(arr[-2]),model_para_files[i]]                        #转化为int格式是为了排序
                model_para_files[i]=arr
            
            model_para_files=sorted(model_para_files)
                
            if not os.path.exists(self.args.optimal_model_dir):
                os.mkdir(self.args.optimal_model_dir)
                    
            #开始验证每个模型
            best_f1=0
            for fl_ in model_para_files:
                print('Validate *_{}_{}.model.'.format(fl_[0],fl_[1]))
                try:
                    for model_name in self.train_models:
                        fl_tmp=os.path.join(
                            '../nats_results', model_name+'_{}_{}.model'.format(fl_[0],fl_[1]))
                        self.train_models[model_name].load_state_dict(
                            torch.load(fl_tmp, map_location=lambda storage, loc:storage))
                except:
                    print('Model can not be load!!!')
                    continue
                        
                val_batch=create_batch_file(path_data=self.args.data_dir,
                                            path_work='../nats_results',
                                            is_shuffle=False,
                                            fkey_=self.args.task,
                                            file_=self.args.file_dev,
                                            batch_size=self.args.batch_size)
                print('The number of validation batches(Dev): {}.'.format(val_batch))
                    
                val_results=[]
                for batch_id in range(val_batch):
                    start_time=time.time()
                    self.build_batch(batch_id)
                    self.test_worker()              #在 test_data 中记录 ‘text_uae’, 'text_reg', 'aspweight', 'label'
                    val_results+=self.test_data     #记录所有 val_results
                    self.test_data=[]
                    end_time=time.time()
                    show_progress(batch_id+1, val_batch,
                                 str(end_time-start_time)[:8]+'s')
                print()                                                      #show_progress的结尾会输出一个'\r'
                
                f1=self.evaluate_worker(val_results)
                print('Best F1 score:{},   current F1 score:{}.'.format(best_f1, f1))
                if f1 > best_f1:
                    for model_name in self.train_models:
                        #保存模型
                        fmodel=open(os.path.join(
                            self.args.optimal_model_dir, '{}.model'.format(model_name)), 'wb')                    
                        torch.save(
                            self.train_models[model_name].state_dict(), fmodel)
                        fmodel.close()
                    #更新best f1
                    best_f1=f1
                    
                    
    ##需要重写
    ##用于抽取属性词
    def aspect_worker(self):
        raise NotImplementedError        
        
    
    ##用于测试模型
    def test(self):
        self.build_vocabulary()                  # 该方法已经把 pretrained vector 导入到了类的属性中了
        self.build_models()
        pprint(self.base_models)
        pprint(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
            
        _nbatch=create_batch_file(path_data=self.args.data_dir,
                                  path_work='../nats_results',
                                  is_shuffle=False, 
                                  fkey_=self.args.task, 
                                  file_=self.args.file_test, 
                                  batch_size=self.args.batch_size)
        
        print('The number of samples(test): {}.'.format(_nbatch))
        
        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
            
        with torch.no_grad():            
            if self.args.use_optimal_model:
                for model_name in self.train_models:
                    model_path=os.path.join(
                        self.args.optimal_model_dir,'{}.model'.format(model_name)) 
                    self.train_models[model_name].load_state_dict(
                        torch.load(model_path, map_location=lambda storage, loc:storage))
            else:
                arr=re.split('\D', self.args.model_optimal_key)
                model_optimal_key='_{}_{}.model'.format(str(arr[0]),str(arr[1]))
                print('You choose *{} to decode.'.format(model_optimal_key))

                for model_name in self.train_models:
                    model_path=os.path.join(
                            '../nats_results', model_name+model_optimal_key)
                    self.train_models[model_name].load_state_dict(torch.load(
                            model_path, map_location=lambda storage, loc:storage))
                
            start_time=time.time()
            out_file=os.path.join(
                    '../nats_results', self.args.file_output)
                
            fout=open(out_file, 'w')
            self.aspect_worker()
            for batch_id in range(_nbatch):
                self.build_batch(batch_id)
                self.test_worker()
                for itm in self.test_data:
                    json.dump(itm, fout)
                    fout.write('\n')
                self.test_data=[]
                end_time=time.time()
                show_progress(batch_id+1, _nbatch, str((end_time-start_time)/3600)[:8]+'h')
            fout.close()
            print()
                
                
    

