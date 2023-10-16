#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
import os
import re
import pandas as pd
import json
import openpyxl

from utils import show_progress, is_Chinese


# In[2]:


#处理中文的方法
#包括：句子切分、去标点、去停词、
class preprocess_chinese:
    def __init__(self):
        self.record = []
        self.stopwords = []
        
    #导入停用词
    def load_stopwords(self, filename = '../停用词表_w2v/停用词库.txt'):
        f_in = open(filename, 'r', encoding = 'utf-8')
        for line in f_in:
            self.stopwords.append(line.strip())
        f_in.close()
        
    #移除非中文词
    #输入: 中文句子;
    #输出: 中文句子。
    def remove_nonchinese(self, sentence):
        precessed_sentence = ''
        for char in sentence:
            if is_Chinese(char):
                precessed_sentence += char
        return precessed_sentence        
    
    #从一个句子里面去除停用词
    #输入：句子
    #输出：去除停用词之后的 list
    def remove_stopwords(self, sentence):
        if len(self.stopwords) == 0:
            self.load_stopwords()        
        segmentation = jieba.lcut(sentence)
        removed_sw_list = [word for word in segmentation if not word in self.stopwords]        
        return removed_sw_list
    
    #对一个爬下来的句子进行处理
    #split: 正则表达式（方括号表达式，用于句子切分）
    #处理步骤：句子切分; 保留中文词; 分词并去停词。 最后会返回一个分完词的 list
    #输入: 一个句子
    #输出: list(list), list(list); 切分好的句子 以及 切分并处理好的句子
    def process_one_sentence(self, ori_sen, split = None):
        if split == 'default' or split is None:
            split = '[。？！；：]'
        sentences = re.split(split, ori_sen)
        
        pre_sentences = []
        for sentence in sentences:
            sentence = self.remove_nonchinese(sentence)
            pre_sentences.append(self.remove_stopwords(sentence))
        return sentences, pre_sentences
        
    #处理爬虫爬下来的文本
    #需要读取 excel 文件：确保 excel 是的文件名是 ‘../爬虫及其结果/{productId}.xlsx’
    #sentence_title: excel 句子保存列名
    def process_one_product(self, productId, split = None, sentence_title = '内容'):
        df = pd.read_excel('../爬虫及其结果/{}.xlsx'.format(productId))
        for ori_sen in df[sentence_title]:
            sentences, pre_sentences = self.process_one_sentence(ori_sen = ori_sen, split = split)            
            for sentence, pre_sentence in zip(sentences, pre_sentences):
                if len(pre_sentence) == 0:
                    continue
                record_dict = {"商品id":productId, "原文本":sentence, "处理后的文本":pre_sentence}
                self.record.append(record_dict)
    
    def process_products(self, productIds, split = None, sentence_title = '内容', return_error = True):
        assert type(productIds) == list
        error_list = []
        total = len(productIds)
        for k, productId in enumerate(productIds):
            try:
                self.process_one_product(productId = productId, split = split, sentence_title = sentence_title)                
            except:
                error_list.append(productId)
            show_progress(k, total, message = '{}/{}'.format(k+1, total))
        print()
        
        if return_error:
            return error_list
        else:
            return None
        
    
    #path: 存储目录
    def save_txt(self, path, file_name = 'train.txt'):
        if not os.path.exists(path):
            os.mkdir(path)
        f_path = os.path.join(path, file_name)
        f_out = open(f_path, 'w', encoding = 'utf-8')
        
        assert len(self.record) > 0
        print('开始保存txt...')
        count = 0
        total = len(self.record)
        for record_dict in self.record:
            json.dump(record_dict, f_out, ensure_ascii=False)     #注意：需要用json.loads解码, ensure_ascii可以避免乱码!
            f_out.write('\n')
            count += 1        
            show_progress(count, total)
            
        f_out.close()
        print()
        print('\n存储完成，共{0}条记录'.format(count))
        
    #path: 存储目录
    def save_excel(self, path, file_name = 'train.xlsx'):
        if not os.path.exists(path):
            os.mkdir(path)
        f_path = os.path.join(path, file_name)
        assert len(self.record) > 0
        
        book = openpyxl.Workbook()
        sheet = book.create_sheet(title='训练数据', index=0)
        column = ['序号', '商品id', '原文本', '处理后的文本']
        sheet.append(column)
        
        print('开始保存excel...')
        total = len(self.record)
        for k, record in enumerate(self.record):
            row = [str(elem) for elem in record.values()]
            row = [k+1] + row
            sheet.append(row)
            show_progress(k+1, total)
            
        book.save(f_path)
        book.close()
        print()
        print('\n存储完成，共{0}条记录'.format(k+1))
        
    def save(self, path, save_excel = True, save_txt = True, 
             excel_name = 'train.xlsx', txt_name = 'train.txt'):  
        if save_excel:
            self.save_excel(path = path, file_name = excel_name)
        if save_txt:
            self.save_txt(path = path, file_name = txt_name)
        


# In[ ]:


if __name__ == '__main__':
    
    id_file = '../爬虫及其结果/笔记本id.txt'
    f = open(id_file, 'r')
    productIds = []
    for line in f:
        ids = line.strip().replace(',', '').split()
        productIds += ids
    f.close()

    pre = preprocess_chinese()
    error_list = pre.process_products(productIds)   
    pre.save('../data')
    print(error_list)




