#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import json
import os
import numpy as np
from sklearn.metrics import classification_report


# In[ ]:


##返回 sscl 模型的 precision, recall, f1-score, accuracy, macro avg, weighted avg
def evaluate_sscl_classification(args):
    aspect_label = []
    fl_ = open(os.path.join('../cluster_results', 'aspect_mapping.txt'), 'r')
    for line in fl_:
        aspect_label.append(line.split()[1])
    fl_.close()
    
    ignore_type = ['nomap']
    if not args.none_type:
        ignore_type.append('none')    
        
    tmp = {wd:-1 for wd in aspect_label if wd not in ignore_type}
    label = {}
    for k, wd in sorted(list(tmp)):
        label[wd] = k
    
    fl_ = open(os.path.join('../cluster_results', args.file_output), 'r')
    pred = []
    gold = []
    for line in fl_:
        itm = json.loads(line)
        arr = np.argsort(itm['aspect_weight'])[::-1]
        for k in arr:
            if not aspect_label[k] in ignore_type:
                pp = aspect_label[k]
                break
        pred.append(label[pp])
        try:
            lb = itm['label']
            gold.append(label[lb])
        except:
            lb = itm['label'].split(',')
            if pp in lb:
                gold.append(label[pp])
            else:
                gold.append(label[lb[0]])
        
    fl_.close()
    
    print(classification_report(
        gold, pred, target_names = list(label), digit = 3))


from utils import show_progress, create_batch_file
from modelSSCL import modelSSCL
from pprint import pprint
import glob
import os
import re
import shutil
import json
import torch
import time


class eval_model(modelSSCL):
    def __init__(self, args):
        super().__init__(args=args)
        self.create_file = False
        self.n_batch = None

    def build_pipe(self):
        attn_pos, ctx_pos = self.build_encoder(
            self.batch_data['pos_sen_var'], self.batch_data['pos_pad_mask'])

        asp_weight = self.train_models['asp_weight'](ctx_pos)  # (b_size, n_aspects)
        asp_weight = torch.softmax(asp_weight, dim=1)

        return attn_pos, asp_weight

    def test_worker(self):
        attn_, asp_weights = self.build_pipe()
        attn_ = attn_.data.cpu().numpy()
        asp_weights = asp_weights.data.cpu().numpy()

        for k in range(len(self.batch_data['uae_tokens'])):
            out = {}
            out['处理后的文本'] = self.batch_data['uae_tokens'][k]
            out['原文本'] = self.batch_data['reg_tokens'][k]
            out['asp_weight'] = np.round(asp_weights[k], 2).tolist()
            out['attn_weight'] = np.round(attn_[k][:len(self.batch_data['uae_tokens'][k])], 2).tolist()  # 只记录非 <pad> 部分
            out['label'] = self.batch_data['label'][k]
            self.test_data.append(out)

    ##把 test 数据中每个类别的 sentence 输出到文件
    def create_data_file(self):
        nats_dir = '../nats_results'
        print('building data...')
        # 构建数据文件（来源于train.txt）
        n_batch = create_batch_file(
            path_data=self.args.data_dir,
            path_work=nats_dir,
            is_shuffle=False,
            fkey_='top_sentence',
            file_='train.txt',
            batch_size=self.args.batch_size,
        )
        self.n_batch = n_batch
        self.create_file = True
        return n_batch

    # 默认 model_id 为最后一步的 model
    # model_id: str '{epoch_id}, {batch_id}' or list: [epoch_id, batch_id]
    # 测试结果输出到文件： {nats_dir}/{file_top_sentence}
    def reporter(self, model_id=None):
        nats_dir = '../nats_results'
        file_top_sentence = 'top_sentence.txt'
        out_file = os.path.join(nats_dir, file_top_sentence)

        print('building vocabulary...')
        self.build_vocabulary()
        print('building models...')
        self.build_models()
        print('initializing base models...')

        if len(self.base_models) > 0:
            self.init_base_model_params()
        if not self.create_file:
            n_batch = self.create_data_file()
        else:
            n_batch = self.n_batch

        # 开始测试
        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        with torch.no_grad():
            # 导入模型 (train)
            if model_id is None:
                model_ids = []
                model_files = glob.glob(
                    os.path.join(nats_dir, 'aspect_embedding*.model'))
                for model_file in model_files:
                    model_id = re.split("_|\.", model_file)
                    batch_id = int(model_id[-2])
                    epoch_id = int(model_id[-3])
                    model_ids.append([epoch_id, batch_id])
                model_ids = sorted(model_ids)
                epoch_id, batch_id = model_ids[-1]

            else:
                try:
                    epoch_id, batch_id = model_id
                except:
                    epoch_id, batch_id = model_id.split(',')

            print('You choice *_{}_{}.model to decode.'.format(epoch_id, batch_id))

            print('initializing train models...')
            for model_name in self.train_models:
                model_path = os.path.join(
                    nats_dir, model_name + '_{}_{}.model'.format(str(epoch_id), str(batch_id)))
                self.train_models[model_name].load_state_dict(
                    torch.load(model_path, map_location=lambda storage, loc: storage))

            pprint(self.base_models)
            pprint(self.train_models)

            start_time = time.time()
            fout = open(out_file, 'w', encoding = 'utf-8')
            print('testing...')
            for batch_id in range(n_batch):
                self.build_batch(batch_id)
                self.test_worker()
                for itm in self.test_data:
                    json.dump(itm, fout, ensure_ascii = False)
                    fout.write('\n')
                self.test_data = []
                end_time = time.time()
                show_progress(batch_id + 1, n_batch, str((end_time - start_time) / 3600)[:8] + 'h')

            print()


            fout.close()
            print('finish testing.')
            print()