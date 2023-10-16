#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np

import os
import json
from End2EndAspDecBase import End2EndAspDecBase

from sklearn.metrics import f1_score
from torch.autograd import Variable
from utils import load_vocab_pretrain


# In[3]:


##需要重写的方法：
#    build_pipe(); get_embedding_weights(); 
#    build_models()
#    app_worker();
##已重写的方法：
#    build_vocabulary(); build_optimizer(params); build_batch(batch_id); build_pipelines(); 
#    test_worker(); aspect_worker(); evaluate_worker(input_)
class modelSSCLBase(End2EndAspDecBase):
    ##args:
    # device(str, 设备), learning_rate(float, 学习率), task(str, 任务名称), batch_size(int, batch大小)
    # min_seq_len(int, 最小句子长度), max_seq_len(int, 最大句子长度), checkpoint(int, 记录点 - 每训练到一定程度，会有记录）
    # warmup_step(int, 与学习率调度和正则化占比相关的数值), n_keywords(int, aspect数量), none_type(bool, aspect是否使用none type)
    def __init__(self,args):
        super().__init__(args=args)
        
    ##构建字典，定将相关信息传递到self.batch_data中
    #wordvec的位置：          {word_dir}/{file_wordvec}('../停用词表_w2v/vectors_w2v.npy')
    #vocab的位置：            {word_dir}/{file_vocab}('../停用词表_w2v/vocab.txt')
    #k_means_centroid的位置： {kmeans_dir}/{file_kmeans_centroid}('../kmeans_results/aspect_centroid.txt')
    #batch_data字典中会有：
    #     vocab2id; id2vocab; pretrain_vec; vocab_size; aspect_vec; n_aspects
    def build_vocabulary(self):
        word_dir='../停用词表_w2v'
        kmeans_dir='../kmeans_results'
        file_wordvec='vectors_w2v.npy'
        file_vocab='vocab.txt'
        file_kmeans_centroid='aspect_centroid.txt'
        
        file_pretrain_vocab=os.path.join(word_dir, file_vocab)
        file_pretrain_vec=os.path.join(word_dir, file_wordvec)
        
        vocab2id, id2vocab, pretrain_vec=load_vocab_pretrain(
            file_pretrain_vocab, file_pretrain_vec)
        
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_vec'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        
        print("The size of vocabulary :{}.".format(vocab_size))
        
        aspect_vec=np.loadtxt(os.path.join(
            kmeans_dir, file_kmeans_centroid), dtype=float)
        aspect_vec=torch.FloatTensor(aspect_vec).to(self.args.device)
        n_aspects=aspect_vec.shape[0]
        
        self.batch_data['aspect_centroid'] = aspect_vec
        self.batch_data['n_aspects'] = n_aspects
        
    
    def build_optimizer(self, params):
        optimizer=torch.optim.Adam(params, lr=self.args.learning_rate)
        return optimizer
    
    
    ##在batch_data中添加数据
    #     添加的记录包括:uae_tokens（预处理的文本）; reg_tokens (原文本) ; pos_sen_var(填充后的句子id) ;
    #                   pos_pad_mask(padding mask) ; label(标签) ; neg_examples(负样本（填充后的句子id、padding mask）)
    #     数据地址:'../nats_results/batch_{fkey_}_{batch_size}/{batch_id}'
    #     tips: 每进行一个 batch 就保存一次 aspect_words
    def build_batch(self, batch_id):
        vocab2id=self.batch_data['vocab2id']
        
        path_=os.path.join('..', 'nats_results')
        fkey_=self.args.task
        batch_size=self.args.batch_size
        file_=os.path.join(
            path_, 'batch_{}_{}'.format(fkey_, str(batch_size)), str(batch_id))
        
        sen_text=[]
        uae_tokens=[]
        reg_tokens=[]
        labels=[]
        asp_mapping=[]
        sen_text_len=0
        
        fp=open(file_, 'r', encoding = 'utf-8')
        for line in fp:
            itm=json.loads(line)
            senid=[                                                                  #由索引表示的句子
                vocab2id[wd] for wd in itm['处理后的文本']
                if wd in vocab2id]
            if len(senid) < self.args.min_seq_len and fkey_[-5:] == 'train':
                continue
            sen_text.append(senid)                                                   #由索引表示的句子组成的集合
            uae_tokens.append(itm['处理后的文本'])
            reg_tokens.append(itm['原文本'])
            if len(senid) > sen_text_len:
                sen_text_len=len(senid)                                              #统计最长句子
            try:
                labels.append(itm['label'].lower())
            except:
                labels.append('none')
        fp.close()
        
        sen_text_len = min(sen_text_len, self.args.max_seq_len)
        sen_text = [                                                                 #填充句子
            sen[:sen_text_len] + 
            [vocab2id['<pad>'] for _ in range(sen_text_len - len(sen))]
            for sen in sen_text]
        
        sen_text_var = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)
        sen_pad_mask = Variable(torch.LongTensor(
            sen_text)).to(self.args.device)

        sen_pad_mask[sen_pad_mask != vocab2id['<pad>']] = -1       #代码写错的地方：这一行和下一行位置颠倒会导致结果不一样！
        sen_pad_mask[sen_pad_mask == vocab2id['<pad>']] = 0
        sen_pad_mask = sen_pad_mask * (-1)
        
        self.batch_data['uae_tokens'] = uae_tokens
        self.batch_data['reg_tokens'] = reg_tokens
        self.batch_data['pos_sen_var'] = sen_text_var
        self.batch_data['pos_pad_mask'] = sen_pad_mask
        self.batch_data['label'] = labels
        
        if self.args.task[-5:] == 'train':
            neg_examples = []
            neg_text=sen_text
            
            for k in range(len(sen_text)):
                neg_text = neg_text[1:] + [neg_text[0]]
                neg_text_var = Variable(torch.LongTensor(
                    neg_text)).to(self.args.device)
                neg_pad_mask = Variable(torch.LongTensor(
                    neg_text)).to(self.args.device)
                neg_pad_mask[neg_pad_mask == vocab2id['<pad>']] = 0
                neg_pad_mask[neg_pad_mask != vocab2id['<pad>']] = -1
                neg_pad_mask = neg_pad_mask * (-1)
                neg_examples.append(
                    [neg_text_var, neg_pad_mask])
            self.batch_data['neg_examples'] = neg_examples
            
            if batch_id % self.args.checkpoint == 0:
                self.aspect_worker()
    
    
    ##需要重写
    ##模型主要前向传播的算法
    #返回：loss和asp_vec
    def build_pipe(self):
        raise NotImplementedError
        
        
    ##正交正则化方法，它会加到loss上
    #      公式：||AA.T-I||
    #      正则化生成的cross_loss在loss中的比例会随训练的进行减小
    def build_pipelines(self):
        #aspect_vec:(batch_size,n_aspect, emb_dim)
        loss, aspect_vec=self.build_pipe()
        
        aspect_norm=aspect_vec / (aspect_vec.norm(p=2, dim=-1, keepdim=True)+1e-20)
        #cross:(b_size,n_aspect,n_aspect)
        cross = aspect_norm @ aspect_norm.transpose(1,2)
        diag = torch.eye(cross.size(1)).to(self.args.device)
        diag = diag.squeeze(0).expand_as(cross)
        diff = cross - diag
        loss_cross = diff.norm(p=2)
        #report
        if 'loss_without_reg' in self.args.report:
            self.report_dict['loss_without_reg'] = np.around(loss.data.cpu().numpy(), 4).tolist()
        if 'loss_reg' in self.args.report:
            self.report_dict['loss_reg'] = np.around(loss_cross.data.cpu().numpy(), 4).tolist()

        return loss + loss_cross*(0.1 + self.args.warmup_step/self.global_steps)
    
    
    ##在属性self.test_data中添加记录（根据self.batch_data），方便在测试时访问：
    #            self.test_data是一个列表，其中的每一条记录都是一个字典，对应着batch_data中的一条数据
    #            添加的记录包括:'text_uae'(预处理后的原文本); 'text_reg'(原文本); 'aspect_weight'(aspect注意力权重); 'label'(标签)
    def test_worker(self):
        asp_weights = self.build_pipe()
        asp_weights = asp_weights.data.cpu().numpy().tolist()
        
        for k in range(len(self.batch_data['uae_tokens'])):
            out={}
            out['text_uae'] = self.batch_data['uae_tokens'][k]
            out['text_reg'] = self.batch_data['reg_tokens'][k]
            out['asp_weight'] = asp_weights[k]
            out['label'] = self.batch_data['label'][k]
            self.test_data.append(out)    
                       
            
    ##需要重写                   
    ##获取词向量(返回emb,asp_emb)  
    def get_embedding_weights(self):
        raise NotImplementedError
        
    ##找到与aspect embedding最接近的topk的单词，并写入文件：
    #         文件名：train:'../nats_results/sccl_aspect_keywords.txt'
    #               test:'../nats_results/test_sccl_aspect_keywords.txt'
    def aspect_worker(self):
        #emb:(v_size, emb_dim)
        #asp_emb:(n_aspect, emb_dim)
        emb, asp_emb=self.get_embedding_weights()
        
        emb=emb.unsqueeze(0)
        asp_emb=asp_emb.unsqueeze(0)
        #score:(n_aspect, v_size)
        score=asp_emb @ emb.transpose(1,2)
        score=score.squeeze(0)
        
        #top_id:(n_aspect, k)
        top_idx=score.topk(
            k=self.args.n_keywords, dim=1)[1].cpu().numpy().tolist()
        
        output=[]
        id2vocab=self.batch_data['id2vocab']
        for idx in top_idx:
            out=[]
            for wd in idx:
                if wd in id2vocab:
                    out.append(id2vocab[wd])
            output.append(out) 
            
        if self.args.task[-5:] == 'train':
            fout=open(os.path.join(
                '../nats_results', 'sscl_aspect_keywords.txt'), 'w')
            for itm in output:
                fout.write('{}\n'.format(' '.join(itm)))
            fout.close()
        else:
            fout=open(os.path.join(
                '../nats_results', 'test_sscl_aspect_keywords.txt'), 'w')
            for itm in output:
                fout.write('{}\n'.format(' '.join(itm)))
            fout.close()
            
        
    ##用于计算f1_score,需要已经标注好的aspect_label文件
    #   input_是一个列表，其中有很多字典（每个字典对应一个uae_tokens），每个字典有两个item，他们的key分别为：‘aspect_weight’和‘label’
    def evaluate_worker(self, input_):
        #文件中会包含手工识别的类别（有顺序，包括了none）
        label_file='../nats_results/aspect_mapping.txt'
        fl_ = open(label_file, 'r')
        aspect_label = []
        for line in fl_:
            aspect_label.append(line.split()[1]) #aspect_label:list -aspect_id to aspect
        #忽略掉'非'的类别(没有手工识别的类别)
        ignore_type=['nomap']
        if not self.args.none_type:             #args.none_type 为 True 时, 'none'不计入
            ignore_type.append('none')
        tmp = {wd: -1 for wd in aspect_label if not wd in ignore_type}
        #label中会仅仅包含手工识别的类别（非'none'）
        label = {}
        for k, wd in enumerate(sorted(list(tmp))):
            label[wd] = k                      #label(去除'none'之后的aspect_label):dict -aspect to aspect_id
            
        pred=[]                                #size=n, n=len(input_)
        gold=[]
        for itm in input_:
            #arr是一个有序索引
            #arr:(n_aspect,)
            arr=np.argsort(itm['aspect_weight'])[::-1]
            for k in arr:
                if not aspect_label[k] in ignore_type:
                    pp = aspect_label[k]       #找到最大att_score对应的索引
                    break
            pred.append(label[pp])             #预测：aspect_id组成的list
            try:
                gold.append(label[itm['label']])
            except:
                lb=itm['label'].split(',')
                if pp in lb:
                    gold.append(label[pp])
                else:
                    gold.append(label[lb[0]])
                    
        return f1_score(gold, pred, average='macro')

        

