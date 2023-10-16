#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from modelSSCLBase import modelSSCLBase


# In[ ]:


##模型前向传播规则：
#  1、build_encoder  - 从输入到生成句子表示的过程；
#                      返回句子注意力 weight 和句子表示；
#  2、build_pipe     - 句子表示的解码过程(解码 aspect_weight)，train 时需要负采样
#                      train:返回模型损失值和 句子的属性表示（方便计算正则化损失）；test:返回 aspect_weight
#  3、build_pipeline - 加入正则化损失
class modelSSCL(modelSSCLBase):
    def __init__(self,args):
        ##args:
        # 层相关:emb_size; device; smooth_factor; distance; if_init_kmeans
        # 模式:  task
        super().__init__(args=args)
    
    ##搭建模型参数：4层有参数：两个embedding,两个linear
    #build_models应该放在build_vocab之后
    def build_models(self):
        self.base_models['embedding']=nn.Embedding(
            self.batch_data['vocab_size'], 
            self.args.emb_size
        ).to(self.args.device)
        
        self.train_models['aspect_embedding']=nn.Embedding(
            self.batch_data['n_aspects'], 
            self.args.emb_size, 
            padding_idx=0
        ).to(self.args.device)
        
        self.train_models['attn_kernel']=nn.Linear(
            self.args.emb_size, 
            self.args.emb_size
        ).to(self.args.device)
        
        self.train_models['asp_weight']=nn.Linear(
            self.args.emb_size, 
            self.batch_data['n_aspects']
        ).to(self.args.device)
        
        
    def init_base_model_params(self):
        emb_para=torch.FloatTensor(
            self.batch_data['pretrain_vec']).to(self.args.device)
        self.base_models['embedding'].weight = nn.Parameter(emb_para)
        
        if self.args.task[-5:] == 'train' and self.args.if_init_kmeans:
            aspect_para = self.batch_data['aspect_centroid']
            aspect_para = nn.Parameter(aspect_para)
            self.train_models['aspect_embedding'].weight = aspect_para
            
    ##从输入到生成句子表示的过程：
    #     主要是attention机制
    #input_,mask_:(batch_size, seq_len)
    #emb_,emb_trn:(batch_size, seq_len, emb_size)
    #emb_avg,enc_:(batch_size, emb_size) - self attention 的 query
    #attn_:(batch_size, seq_len) - 句子单词的 attention weight
    #ctx_:(batch_size, emb_size) - 句子的表示 attention 权重分配给 word embedding 之后的结果
    def build_encoder(self, input_, mask_):
        
        with torch.no_grad():
            emb_ = self.base_models['embedding'](input_)
        emb_ = emb_ * mask_.unsqueeze(2)                    #(batch_size, seq_len, emb_size)
        
        emb_avg = torch.sum(emb_, dim=1)
        norm = torch.sum(mask_, dim=1, keepdim=True) + 1e-20        #(batch_size, 1)
        enc_ = emb_avg.div(norm.expand_as(emb_avg))                 #(batch_size, emb_size) self_attention 的 query
        
        emb_trn = self.train_models['attn_kernel'](emb_)   #(batch_size, seq_len, emb_size) attention score 计算中，embedding的特征变换
        attn_ = enc_.unsqueeze(1) @ emb_trn.transpose(1,2) #(batch_size, 1, seq_len)
        attn_ = attn_.squeeze(1)
        attn_ = self.args.smooth_factor * torch.tanh(attn_)
        attn_ = attn_.masked_fill(mask_ == 0, -1e20)
        attn_ = torch.softmax(attn_, dim=1)                #(batch_size, seq_len)
        
        #分配注意力
        ctx_ = attn_.unsqueeze(1) @ emb_                   #(batch_size, 1, emb_size)
        ctx_ = ctx_.squeeze(1)
        
        return attn_, ctx_
    
    
    ##计算vec1和vec2之间的余弦相似度
    #   vec1/vec2:(batch_size, emb_size)
    #   score:(batch_size, )
    def compute_distance(self, vec1, vec2):
        vec1 = vec1 / (vec1.norm(p=2, dim=1, keepdim=True) + 1e-20)
        vec2 = vec2 / (vec2.norm(p=2, dim=1, keepdim=True) + 1e-20)
        
        if self.args.distance == 'cosine':
            score = vec1.unsqueeze(1) @ vec2.unsqueeze(2)      #(batch_size, 1, 1)
            score = score.squeeze(1).squeeze(1)
            
        return score
    
    
    ##句子表示的解码过程，训练状态需要负采样
    # 返回：train: loss_cdae, asp_emb; 非train: asp_weight
    # ctx_pos:(batch_size, emb_size)
    # asp_weight:(batch_size, n_aspects)
    # asp:(n_aspects, )
    # asp_enc:(batch_size, emb_size)
    # score_pos:(batch_size,)
    # score_neg:(batch_size, batch_size)
    def build_pipe(self):
        b_size=self.batch_data['pos_sen_var'].size(0)
        attn_pos, ctx_pos = self.build_encoder(
            self.batch_data['pos_sen_var'], self.batch_data['pos_pad_mask'])
        
        #打印attention值(已注释)
        ''' if 'attention' in self.args.monitor \
                and self.global_steps >= 15000 \
                and self.global_steps % 3000 ==0:
            print()
            for k in range(len(self.batch_data['reg_tokens'])):
                outw = np.around(attn_pos[k].data.cpu().numpy().tolist(), 4)
                outw = outw.tolist()
                outw = outw[:len(self.batch_data['uae_tokens'][k].split())]
                print(outw)
                break'''

        
        asp_weight = self.train_models['asp_weight'](ctx_pos)      #(b_size, n_aspects)
        asp_weight = torch.softmax(asp_weight, dim=1)              #(batch_size, n_aspects)

        #report
        if 'asp_weight_norm' in self.args.report and self.args.task == 'train':
            asp_w = asp_weight.data.cpu().numpy()
            asp_weight_norm = asp_w * asp_w                                  #(batch_size, n_aspect)
            asp_weight_norm = np.mean(np.sum(asp_weight_norm, axis = -1)).tolist()
            self.report_dict['asp_weight_norm'] = asp_weight_norm

        #打印asp_weight值(已注释)
        '''if 'asp_weight' in self.args.monitor \
                and self.global_steps >= 15000 \
                and self.global_steps % 3000 == 0:
            out_asp = np.around(asp_weight[0].data.cpu().numpy().tolist(), 2)
            out_norm = np.dot(out_asp, out_asp.transpose()).tolist()
            out_asp=out_asp.tolist()
            print(out_norm, '\n', out_asp)'''
        
        if self.args.task[-5:] == 'train' :
            asp = torch.LongTensor(range(self.batch_data['n_aspects'])) #(n_aspects, )
            asp = Variable(asp).to(self.args.device)
            asp = asp.unsqueeze(0).repeat(b_size, 1)
            asp_emb = self.train_models['aspect_embedding'](asp)        #(b_size, n_aspects, emb_size)
            asp_enc = asp_weight.unsqueeze(1) @ asp_emb                 #(b_size, 1, emb_size) - 分配注意力
            asp_enc = asp_enc.squeeze(1)                                #(b_size, emb_size) - aspect表示
            
            score_pos = self.compute_distance(asp_enc, ctx_pos)         #(b_size, )
            
            score_neg_arr = []
            for itm in self.batch_data['neg_examples']:                 #
                _, ctx_neg = self.build_encoder(itm[0], itm[1])         #(b_size, emb_size)
                score_neg = self.compute_distance(asp_enc, ctx_neg)     #(b_size, ) 注意：这里的score_neg并不是一个data计算出来的,
                                                                        #                而是由batch中各个data计算出来的
                score_neg_arr.append(torch.exp(score_neg))              #list: [b_size * (b_size, )]
            
            score_neg = torch.cat(score_neg_arr, 0).view(-1, score_pos.size(0)) #(b_size, b_size)
            score_neg = score_neg.contiguous().transpose(0, 1)          #(b_size, b_size) 转置过来之后dim=1对应每个batch的负采样
            score_neg = torch.sum(score_neg, -1)                        #(b_size, )
            loss_cdae = torch.mean(-score_pos + torch.log(score_neg))

            return loss_cdae, asp_emb
        
        else:
            return asp_weight
        
    
    ##获取embedding参数(包括embedding和aspect_embedding)
    def get_embedding_weights(self):
        
        emb = self.base_models['embedding'].weight
        asp_emb = self.train_models['aspect_embedding'].weight
        
        return emb, asp_emb
    
    

