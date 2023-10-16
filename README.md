# SSCL-SAI
The code and data for the paper "A novel self-supervised contrastive learning-based sentence-level attribute induction method for online satisfaction evaluation".

## Requirements
- Python 3.8.8
- torch==1.9.0
- numpy==1.20.3
- gensim==4.0.1
- sklearn==0.0
- jieba==0.42.1

## Quick Start
### Preprocessing
```
./源代码/preprocess.py
```

### Train
```
./源代码/main.ipynb
```

### Sentence Classification
```
./源代码/模型评估.ipynb
```

## Folders
### 停用词表_w2v
This folder saves the stop words used in our paper. Also, the trained word vectors will saved in this folder.
#### Files
```
停用词库.txt: The stop words that downloaded form CSDN
```

### 爬虫及其结果
The crawler.
#### Files
```
crawler.py: The code of crawler that could crawl the reviews from Jd.com.
笔记本ID.txt: The producr IDs of target laptops
{product_id}.xlsx: Reviews of one laptop
```

### 源代码
The main code for SSCL-SAI.
#### Files
```
preprocess.py.py: Preprocess the data with Sentence Segmentation, Remove Non-Chinese words, Word Segmentation and Remove Stop words.
word2vec.py: Train the word vertors.
kmeans.py: Generate the centroids of word vecters.
End2EndBase.py: Body of the model.
main.ipynb: Run the word2vec, kmeans and the model accordingly
evaluation.py: Classify the sentence
模型评估,ipynb: Implement SSCL-SAI and generate excel files
utils.py: Some tools for model and plotting.
```

### 对比实验
This Folder shows the result of experiments in our paper
#### Files
```
作图: The variation of loss.
kmeans: The result of kmeans.
report: Some parameters when the model is training.
excel: The result of sentence-level classification.
```
