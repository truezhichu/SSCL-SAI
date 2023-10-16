#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import Word2Vec
import json
import os
import numpy as np


# In[2]:


#args:
#    emb_size; window; min_count; workers
class run_w2v:
    def __init__(self, args):
        self.args = args
        self.sentences = []
        self.model = None
        
    def load_sentence(self, path = '../data/train.txt'):
        sentences = []
        f_in = open(path, 'r', encoding = 'utf-8')
        for line in f_in:
            text_dict = json.loads(line.strip())
            sentence = text_dict['处理后的文本']
            sentences.append(sentence)
        self.sentences = sentences
        
    def train_model(self, save_path = '../停用词表_w2v'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        if len(self.sentences) == 0:
            self.load_sentence(path = '../data/train.txt')
        args = self.args
        print('开始训练词向量。。。')
        print(f'共有{len(self.sentences)}个句子。')
        model = Word2Vec(sentences = self.sentences, vector_size = args.emb_size,
                         window = args.window, min_count = args.min_count,
                         workers = args.workers, sg = 1)
        
        if save_path:
            file_path = os.path.join(save_path, 'w2v_embedding')
            model.save(file_path)            
        print('训练完成！')    
        self.model = model
        
        return model
    
    #必须保持词表索引和词向量索引顺序一致！
    def save_vocab_and_vectors(self, path = '../停用词表_w2v',
                               vocab_name = 'vocab.txt', vectors_name = 'vectors_w2v'):
        assert not self.model is None
        args = self.args
        if not os.path.exists(path):
            os.mkdir(path)
        
        vocab_path = os.path.join(path, vocab_name)
        print('正在写入字典，格式：(单词 索引)')
        wv = self.model.wv
        vocab_list = wv.index_to_key
        
        f_out = open(vocab_path, 'w')
        for k, word in enumerate(vocab_list):
            line = str(word) + ' ' + str(k) +'\n'
            f_out.write(line)
        f_out.close()
        print('写入完成！')
        
        vectors_path = os.path.join(path, vectors_name)
        emb_vectors = np.zeros([len(vocab_list), args.emb_size])
        print('正在写入词向量(numpy)')
        for k, word in enumerate(vocab_list):
            emb_vectors[k] = wv[word]
        np.save(vectors_path, emb_vectors)
        print('写入完成！')
    
    


# In[3]:


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--window', default = 5, type = int,
                        help = 'window size')

    parser.add_argument('--min_count', default = 10, type = int, 
                        help = 'the minimum count of word')

    parser.add_argument('--workers', default = 8, type = int, 
                        help = 'workers')

    parser.add_argument('--emb_size', default = 128, type = int, 
                        help = 'embedding size')

    args = parser.parse_args([])


# In[4]:


if __name__ == '__main__':
    runner = run_w2v(args)
    model = runner.train_model()
    runner.save_vocab_and_vectors()

