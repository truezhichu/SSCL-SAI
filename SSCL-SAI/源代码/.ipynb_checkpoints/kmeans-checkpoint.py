#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import json

from gensim.models import Word2Vec
from sklearn.cluster import KMeans


# In[2]:


##args:
# n_clusters(中心数量); kmeans_seeds(初始化种子);
# emb_size(embedding 维度); topn(最临近的单词个数)
class run_kmeans:
    def __init__(self, args):
        self.args = args
        self.emb_matrix = None                                       #(vocab_size, emb_dim)
        self.index_to_word = None
        self.centers = None
        self.wv = None
        
    def load_embedding(self, emb_path = '../停用词表_w2v/vectors_w2v.npy'):
        self.emb_matrix = np.load(emb_path)
        
    def load_w2v(self, w2v_path = '../停用词表_w2v/w2v_embedding'):
        self.wv = Word2Vec.load(w2v_path).wv
        
    def load_vocab(self, vocab_path = '../停用词表_w2v/vocab.txt'):
        index_to_word = {}
        f = open(vocab_path, 'r')
        for line in f:
            word, idx = line.strip().split(' ')
            idx = int(idx)
            index_to_word[idx] = word
        f.close()
        self.index_to_word = index_to_word
        
    def train(self, center_path = '../kmeans_results/aspect_centroid.txt',
              topn_path = '../kmeans_results/topn.txt', topn = 10):
        if self.emb_matrix is None:
            self.load_embedding()
            
        if not os.path.exists('../kmeans_results'):
            os.mkdir('../kmeans_results')
         
        args = self.args
        print('正在训练模型。。。')
        model = KMeans(args.n_clusters, random_state = args.kmeans_seeds)
        model.fit(self.emb_matrix)
        print('训练完成！')
        
        centers = model.cluster_centers_                                 #(n_clusters, emb_dim)
        self.centers = centers
        
        if center_path:
            print('正在保存中心。。。')
            np.savetxt(center_path, centers)
            print('保存完成！')
            
        if topn_path:   
            '''
            weight = np.matmul(centers, self.emb_matrix.transpose())     #(n_clusters, vocab_size)
            if self.index_to_word is None:
                self.load_vocab()
            
            out = []
            order = np.argsort(weight)[::-1][:, :topn]                   #(n_clusters, topn)
            for i in range(args.n_clusters):
                topn_words = [self.index_to_word[idx] for idx in order[i]]
                topn_words = ' '.join(topn_words)
                out.append(topn_words)
            '''
            print('正在保存topn = {} 的临近词。。。'.format(topn))
            if self.wv is None:
                self.load_w2v()
            wv = self.wv
            out = [[] for _ in range(args.n_clusters)]
            for i in range(args.n_clusters):
                for word, k in wv.similar_by_vector(centers[i], topn = topn):
                    out[i].append(word)
                out[i] = ' '.join(out[i])
                
            f_out = open(topn_path, 'w')
            for topn_words in out:
                f_out.write(topn_words)
                f_out.write('\n')
            f_out.close()
        print('保存完成！')
        


# In[3]:


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_clusters', default = 30, type = int,
                        help = 'the number of clusters')

    parser.add_argument('--kmeans_seeds', default = 0, type = int, 
                        help = 'random seeds')

    parser.add_argument('--emb_size', default = 128, type = int, 
                        help = 'embedding size')

    parser.add_argument('--topn', default = 10, type = int, 
                        help = 'topn words')

    args = parser.parse_args([])


# In[4]:


if __name__ == '__main__':
    runner = run_kmeans(args)
    runner.train()

