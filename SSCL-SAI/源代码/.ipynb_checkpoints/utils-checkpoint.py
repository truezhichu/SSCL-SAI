#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import numpy as np

import os
import sys
import shutil

import re
import time
from tqdm import tqdm
import random

import argparse


# In[7]:


#打印进度条：[>>>>>>>>>>>---------] {percent}% {message}
#输入：目前个数、总个数、提示信息
def show_progress(curr_,total_,message=''):
    percent=int(round(float(curr_)/float(total_)*100))
    progress='['+'>'*int(round(percent/4))+'-'*(25-int(round(percent/4)))+']'
    sys.stdout.write('{} {}% {}\r'.format(progress,percent,message))
    sys.stdout.flush()
    


# In[ ]:


##构造数据的batch文件
#输入文件: {path_data}/{file_}
#输出文件: {path_work}/batch_{fkey_}_{batch_size}
#该文件最大数据量: 100,000*batch_size
def create_batch_file(path_data, path_work,
                      is_shuffle, fkey_, file_,
                      batch_size):
    file_name = os.path.join(path_data, file_)
    folder = os.path.join(path_work,
                          'batch_{}_{}'.format(fkey_, str(batch_size)))

    try:
        shutil.rmtree(folder)
        os.mkdir(folder)
    except:
        os.mkdir(folder)

    fin = open(file_name, 'r', encoding='utf-8')
    cnt = 0
    corpus_arr = []

    print('loading data...')
    for line in fin:
        corpus_arr.append(line)
        if len(corpus_arr) == 100000 * batch_size:
            if is_shuffle:
                random.shuffle(corpus_arr)

            arr = []
            total = int(np.ceil(len(corpus_arr) / batch_size))
            print('batch number: {}'.format(total))
            print('writing data...')
            for itm in corpus_arr:
                arr.append(itm)
                if len(arr) == batch_size:
                    fout = open(os.path.join(folder, str(cnt)), 'w', encoding = 'utf-8')
                    for sen in arr:
                        fout.write(sen)
                    fout.close()
                    arr = []
                    cnt += 1
                    if cnt % 50 == 0:
                        show_progress(cnt, total)
            print()
            # 写入剩余的arr(不满一个batch)
            if len(arr) > 0:
                fout = open(os.path.join(folder, str(cnt)), 'w', encoding = 'utf-8')
                for sen in arr:
                    fout.write(sen)
                fout.close()
                arr = []
                cnt += 1

            corpus_arr = []

    if len(corpus_arr) > 0:
        if is_shuffle:
            random.shuffle(corpus_arr)

        arr = []
        total = int(np.ceil(len(corpus_arr) / batch_size))
        print('batch number: {}'.format(total))
        print('writing data...')
        for itm in corpus_arr:
            arr.append(itm)
            if len(arr) == batch_size:
                fout = open(os.path.join(folder, str(cnt)), 'w', encoding = 'utf-8')
                for sen in arr:
                    fout.write(sen)
                fout.close()
                arr = []
                cnt += 1
                if cnt % 50 == 0:
                    show_progress(cnt, total)
        print()

        # 写入剩余的arr(不满一个batch)
        if len(arr) > 0:
            fout = open(os.path.join(folder, str(cnt)), 'w', encoding = 'utf-8')
            for sen in arr:
                fout.write(sen)
            fout.close()
            arr = []
            cnt += 1
        corpus_arr = []

    fin.close()

    return cnt
        
        


# In[ ]:


##导入字典和embedding
#读取两个文件：1、单词+索引 (使用' '或'<sec>'分割); 2、根据词频排好序的 word embedding
#返回两个字典：vocab2id, id2vocab; 一个numpy数组：pretrain_vec
def load_vocab_pretrain(file_pretrain_vocab, file_pretrain_vec, pad_tokens=True):
    if pad_tokens:
        vocab2id={'<s>':2, '</s>':3, '<pad>':1, '<unk>':0, '<stop>':4}
        id2vocab={2:'<s>', 3:'</s>', 1:'<pad>', 0:'<unk>', 4:'<stop>'}
        word_pad={'<s>':2, '</s>':3, '<pad>':1, '<unk>':0, '<stop>':4}
    else:
        vocab2id={}
        id2vocab={}
        word_pad={}
        
    pad_cnt=len(vocab2id)
    cnt=len(vocab2id)
    with open(file_pretrain_vocab, 'r') as f:
        for line in f:
            arr=re.split(' ', line.strip())
            if len(arr) == 1:
                arr=re.split('<sec>', line.strip())
            if arr[0] == ' ':
                continue
            if arr[0] in word_pad:
                continue
            arr=arr[0]
            vocab2id[arr]=cnt
            id2vocab[cnt]=arr
            cnt+=1
            
    pretrain_vec=np.load(file_pretrain_vec)
    pad_vec=np.zeros([pad_cnt, pretrain_vec.shape[-1]])
    pretrain_vec=np.vstack((pad_vec, pretrain_vec))
    
    return vocab2id, id2vocab, pretrain_vec


def str2bool(input_):
    if input_.lower() in ('true', 'y', 'yes', '1', 't'):
        return True
    elif input_.lower() in ('false', 'n', 'no', '0', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 应该放到 utils 中去
# 用来判断是否是中文字符
def is_Chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


# In[ ]:


# 比较两个字符串中不同的部分，并返回一个字典，其中表示了不同的部分的位置和内容。
# 要求两个字符串一样长
def detect_char_difference(a, b):
    a_dict = {'pos': [], 'content': []}
    b_dict = {'pos': [], 'content': []}
    a_list = []
    b_list = []
    interval = []
    temp = -10
    for i, (char_a, char_b) in enumerate(zip(a, b)):
        if char_a != char_b:
            temp = i
            interval.append(temp)
            a_list.append(char_a)
            b_list.append(char_b)

        if temp == i - 1:
            add_a = ''.join(a_list)
            add_b = ''.join(b_list)
            interval = (interval[0], interval[-1])
            a_dict['content'].append(add_a)
            b_dict['content'].append(add_b)
            a_dict['pos'].append(interval)
            b_dict['pos'].append(interval)

            a_list = []
            b_list = []
            interval = []

        if i == min(len(a), len(b)) - 1 and len(a_list) > 0:
            add_a = ''.join(a_list)
            add_b = ''.join(b_list)
            interval = (interval[0], interval[-1])
            a_dict['content'].append(add_a)
            b_dict['content'].append(add_b)
            a_dict['pos'].append(interval)
            b_dict['pos'].append(interval)

    return a_dict, b_dict

def run_task(args):
    if args.task == 'word2vec':
        from word2vec import run_word2vec, convert_vectors
        run_word2vec(args)  # 训练词向量，并保存词向量文件
        convert_vectors(args)  # 根据词向量文件，保存 vocab.txt 和 vectors_w2v.npy

    if args.task[:6] == 'kmeans':
        if args.task == 'kmeans':
            from kmeans import run_kmeans, get_cluster_keywords
            run_kmeans(args)  # 训练并保存和 kmeans 相关的文件
            get_cluster_keywords(args)  # 获取 kmeans 中心
        if args.task == 'kmeans-keywords':
            from kmeans import get_cluster_keywords
            get_cluster_keywords(args)

    if args.task[:4] == 'sscl':
        args.device = torch.device(args.device)
        from modelSSCL import modelSSCL
        model = modelSSCL(args)

        if args.task == 'sscl-train':
            model.train()

        if args.task == 'top_sentence':
            from evaluation import eval_model
            model = eval_model(args)
            model.reporter()

        if args.task == 'sscl-validate':
            args.optimal_model_dir = '../nats_results/sscl_models'
            model.validate()

        if args.task == 'sscl-test':
            args.file_output = 'test_sscl_output.json'
            args.optimal_model_dir = '../nats_results/sscl_models'
            model.test()

        '''if args.task == 'sscl-teacher':'''

        if args.task == 'sscl-evaluate':
            from evaluation import evaluate_sscl_classification
            args.file_output = 'test_sscl_output.json'
            evaluate_sscl_classification(args)

        if args.task == 'sscl-clean':
            from glob import glob
            import shutil

            out_dir = '../nats_results/sscl_train_models'

            model_files = glob('../nats_results/*.model')
            if os.path.exists(out_dir) and len(model_files) > 0:
                shutil.rmtree(out_dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            for fl_ in model_files:
                shutil.move(fl_, fl_.replace(
                    'nats_results', 'nats_results/sscl_train_models'))

            shutil.copy('../nats_results/args.pickled',
                        os.path.join(out_dir, 'args.pickled'))
            shutil.copy('../nats_results/sscl_aspect_keywords.txt',
                        os.path.join(out_dir, 'sscl_aspect_keywords.txt'))
            shutil.copy('../nats_results/aspect_mapping.txt',
                        os.path.join(out_dir, 'aspect_mapping.txt'))

            data_files = glob('../nats_results/batch_*')
            for fl_ in data_files:
                shutil.rmtree(fl_)


#作图工具：用途是画 report 的图
import matplotlib.pyplot as plt
import json
import glob
class plot_report():
    def __init__(self):
        self.record_dict = {}
        self.epoch_id = []
        self.steps = 0

    def load_report(self, epoch_id=[], report_folder='../nats_results'):
        assert type(epoch_id) == list
        if len(epoch_id) == 0:
            files = glob.glob(os.path.join(report_folder, 'report_*.txt'))
            for file in files:
                idx = int(re.split('_|\.', file)[-2])
                epoch_id.append(idx)
        if len(epoch_id) == 2 and epoch_id[-1] - epoch_id[0] > 1:
            epoch_id = list(range(epoch_id[0], epoch_id[-1])) + [epoch_id[-1]]
        self.epoch_id = epoch_id

        record_dict = {}
        for i in epoch_id:
            f_report = open(os.path.join(report_folder, 'report_{}.txt'.format(str(i))), 'r')
            for line in f_report:
                report_dict = json.loads(line)
                try:
                    record_dict['epoch'].append(i)
                except:
                    record_dict['epoch'] = [i]
                assert 'step' in report_dict.keys()
                for key, value in report_dict.items():
                    try:
                        record_dict[key].append(value)
                    except:
                        record_dict[key] = [value]
            f_report.close()
        self.record_dict = record_dict
        self.steps = record_dict['step'][-1]
        return record_dict

    def compute_mean(self, start2end, target):
        assert len(start2end) == 2
        assert target in ['asp_weight_norm', 'loss_without_reg', 'loss_reg', 'lr', 'loss']
        start, end = start2end
        cc =0
        start_idx, end_idx = 0,0
        for itm in self.record_dict['step']:
            if itm <=  start:
                start_idx = cc
            if itm <= end:
                end_idx = cc
            cc += 1
        mean = np.mean(np.asarray(self.record_dict[target][start_idx:end_idx+1]))
        return mean



import matplotlib.pyplot as plt
import json
import glob
import os
import numpy as np
import re

# draw:list; start2end:step 的开始到结束； form: 输出形式是 'step', 还是 'epoch'
class plot_report():
    def __init__(self):
        self.record_dict = {}
        self.epoch_id = []
        self.steps = 0

    def load_report(self, epoch_id=[], report_folder='../nats_results'):
        assert type(epoch_id) == list
        if len(epoch_id) == 0:
            files = glob.glob(os.path.join(report_folder, 'report_*.txt'))
            for file in files:
                idx = int(re.split('_|\.', file)[-2])
                epoch_id.append(idx)
        if len(epoch_id) == 2 and epoch_id[-1] - epoch_id[0] > 1:
            epoch_id = list(range(epoch_id[0], epoch_id[-1])) + [epoch_id[-1]]
        epoch_id = sorted(epoch_id)
        self.epoch_id = epoch_id

        record_dict = {}
        for i in epoch_id:
            f_report = open(os.path.join(report_folder, 'report_{}.txt'.format(str(i))), 'r')
            for line in f_report:
                report_dict = json.loads(line)
                try:
                    record_dict['epoch'].append(i)
                except:
                    record_dict['epoch'] = [i]
                assert 'step' in report_dict.keys()
                for key, value in report_dict.items():
                    try:
                        record_dict[key].append(value)
                    except:
                        record_dict[key] = [value]
            f_report.close()
        self.record_dict = record_dict
        self.steps = record_dict['step'][-1]
        return record_dict

    # draw:list; start2end:step 的开始到结束； form: 输出形式是 'step', 还是 'epoch'
    def draw_report(self, draw, form='step', start2end=[], skip=50, epoch_id=[], report_folder='../nats_results',
                    figure_size=(6, 6), xlim=None, ylim=None, save_folder=None):
        self.load_report(epoch_id = epoch_id, report_folder = report_folder)
        assert type(start2end) == list
        if len(start2end) == 0:
            start2end = [0, self.steps]
        assert len(start2end) == 2

        al = set(['asp_weight_norm', 'loss_without_reg', 'loss_reg', 'lr', 'loss'])
        c1 = len(al)
        al.update(draw)
        assert len(al) == c1

        start, end = start2end
        record_dict = self.record_dict
        for key in draw:
            if key == 'step':
                continue

            assert len(figure_size) == 2
            fig_x, fig_y = figure_size
            plt.figure(figsize=(fig_x, fig_y))
            x = np.around(np.asarray(record_dict['step'][start:end:skip]) , 1)
            y = record_dict[key][start:end:skip]
            plt.plot(x, y)
            plt.legend([key])

            if not xlim is None:
                assert len(xlim) == 2
                a1, b1 = xlim
                plt.xlim(a1, b1)

            if not ylim is None:
                assert len(ylim) == 2
                a2, b2 = ylim
                plt.ylim(a2, b2)

            plt.ylabel(key)
            assert form in ['step', 'epoch', 'epoch-step']
            if form == 'epoch':
                n_epoch = len(self.epoch_id)
                step_transfer = np.around(np.linspace(int(x[0]), int(x[-1]), n_epoch), 1)
                plt.xticks(step_transfer, self.epoch_id)
                plt.xlabel('epoch')
                plt.title('epoch - ' + key)
                if not save_folder is None:
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    fl_ = key + '-epoch:' + str(self.epoch_id[0]) + str(self.epoch_id[-1]) + '.png'
                    save_file = os.path.join(
                        save_folder, fl_)
                    plt.savefig(save_file, format='png')

            elif form == 'epoch-step':
                n_epoch = len(self.epoch_id)
                step_transfer = np.around(np.linspace(int(x[0]), int(x[-1]), n_epoch), 1)

                label = []
                for e, s in zip(self.epoch_id, step_transfer):
                    label.append(str(e) + '_' + str(s))

                plt.xticks(step_transfer, label)
                plt.xlabel('epoch-step(k)')
                plt.title('epoch_step(k) - ' + key)
                if not save_folder is None:
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    fl_ = key + ' epoch-' + str(self.epoch_id[0]) + '-' + str(self.epoch_id[-1]) + '.png'
                    print(fl_)
                    save_file = os.path.join(
                        save_folder, fl_)
                    plt.savefig(save_file, format='png')

            else:
                plt.xlabel('step')
                plt.title('step(k) - ' + key)
                if not save_folder is None:
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    save_file = os.path.join(
                        save_folder, '{}-step:{}to{}.png'.format(
                            key, x[0], x[-1]))
                    plt.savefig(save_file, format='png')

            plt.show()


import json
import os
import numpy as np
import pandas as pd


class read_top_sen():
    def __init__(self, n_cluster = 30):
        self.record = {}
        self.n_cluster = n_cluster

    # report_dict.keys(): ['处理后的文本', '原文本', 'asp_weight', 'attn_weight',
    #                     'pre_aspect', 'pre_asp_confidence', 'label']
    def load_record(self):
        print('loading data...')
        nats_folder = '../nats_results'
        file = 'top_sentence.txt'
        f_in = open(os.path.join(nats_folder, file), 'r', encoding = 'utf-8')

        for line in f_in:
            report_dict = json.loads(line)
            for key in report_dict:
                if key in ['asp_weight', 'attn_weight']:
                    report_dict[key] = np.round(report_dict[key], 2)
                    if key == 'asp_weight':
                        self.sort_asp(report_dict)
                try:
                    self.record[key].append(report_dict[key])
                except:
                    self.record[key] = [report_dict[key]]

        f_in.close()

    def sort_asp(self, report_dict):
        select_idx = np.argsort(report_dict['asp_weight'])[-1]
        confidence = report_dict['asp_weight'][select_idx]
        try:
            self.record['pre_aspect'].append(select_idx)
            self.record['pre_asp_confidence'].append(confidence)
        except:
            self.record['pre_aspect'] = [select_idx]
            self.record['pre_asp_confidence'] = [confidence]

    def visit_dict(self, idx):
        dic = {}
        dic['id'] = str(idx)
        dic['pre_aspect'] = self.record['pre_aspect'][idx].tolist()
        dic['pre_asp_confidence'] = self.record['pre_asp_confidence'][idx].tolist()
        dic['处理后的文本'] = self.record['处理后的文本'][idx]
        dic['原文本'] = self.record['原文本'][idx]
        dic['asp_weight'] = self.record['asp_weight'][idx].tolist()
        dic['attn_weight'] = self.record['attn_weight'][idx].tolist()
        dic['label'] = self.record['label'][idx]

        return dic

    ##按照类别写入文件：'../category'
    # 写入内容包括：'pre_aspect','pre_asp_confidence','text_uae','text_reg','attn_weight','asp_weight'
    def create_txt(self, threhold=0.6):
        if len(self.record) == 0:
            self.load_record()
        if not os.path.exists('../txt'):
            os.mkdir('../txt')

        folder = '../txt/category_(threhold-{})'.format(threhold)
        if not os.path.exists(folder):
            os.mkdir(folder)

        total = len(self.record['pre_asp_confidence'])
        for k, confidence in enumerate(self.record['pre_asp_confidence']):
            cc = 0
            if confidence >= threhold:
                file = '{}.txt'.format(self.record['pre_aspect'][k])
                fout = open(os.path.join(folder, file), 'a', encoding = 'utf-8')
                dic = self.visit_dict(k)
                json.dump(dic, fout, ensure_ascii = False)
                fout.write('\n')
                fout.close()
                cc = k

            show_progress(k, total, message='{}/{} {}'.format(k, total, cc))

    def create_excel(self, threhold, min_len=5):
        if len(self.record) == 0:
            self.load_record()
        if not os.path.exists('../excel'):
            os.mkdir('../excel')

        folder = '../excel/category_(threhold-{}, min_len-{})'.format(threhold, min_len)
        if not os.path.exists(folder):
            os.mkdir(folder)

        df = pd.DataFrame(self.record)
        for i in range(self.n_cluster):
            print('writing excel(cluster = {}, threhold = {})'.format(i, threhold))
            file = '{}.xlsx'.format(i)
            df_ = df[(df['pre_aspect'] == i) & (df['pre_asp_confidence'] >= threhold)]
            df_.to_excel(os.path.join(folder, file))