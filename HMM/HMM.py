'''
Author: wbs2788
Date: 2021-10-21 09:56:50
LastEditTime: 2021-10-27 22:49:58
LastEditors: wbs2788
Description: file information
'''

import json
from collections import defaultdict

import numpy as np
from seqeval.metrics import classification_report
from tqdm import tqdm


def viterbi(sentence, pi_mat, trans_mat, obs_mat, labels) -> list:
    """viterbi algorithm
    Args:
        sentence (dict): single sentence
        pi_mat (dict): initialization matrix
        trans_mat (dict): transform matrix
        obs_mat (dict): observe matrix
        labels (list): all labels

    Returns:
        (list): sequence    
    """    
    sentence_cnt = len(sentence)
    label_cnt = len(labels)
    nn = np.zeros((label_cnt, sentence_cnt))
    mm = np.zeros((label_cnt, sentence_cnt))
    probs = np.array([words[sentence[0]] for words in obs_mat])
    seq = [0 for _ in range(sentence_cnt)]
    nn[:, 0] = pi_mat * probs
    for i in range(1,len(sentence)):
        probs = np.array([words[sentence[i]] for words in obs_mat]).reshape((1,-1))
        nn[:, i] = np.max(nn[:, i - 1] * trans_mat.T * probs.T, 1)
        mm[:, i] = np.argmax(nn[:, i - 1] * trans_mat.T, 1)

    seq[-1] = np.argmax(nn[:, sentence_cnt - 1])
    for i in reversed(range(1, sentence_cnt)):
        seq[i - 1] = mm[int(seq[i]), i]
    return seq

with open("数据集/dev.json", encoding='utf-8') as f:
    data = f.readlines()
dev = [json.loads(i) for i in data]

with open("数据集/train.json", encoding='utf-8') as f:
    data = f.readlines()
train = [json.loads(i) for i in data]

dataset = []
for sentence in train:
    dic = {}
    text = sentence['text']
    tag = ['O' for i in range(len(text))]
    for label in sentence['label']:
        for i in sentence['label'][label]:
            for j in sentence['label'][label][i]:
                if label == 'address':
                    tag[j[0]] = 'B-ADDRESS'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-ADDRESS'
                elif label == 'book':
                    tag[j[0]] = 'B-BOOK'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-BOOK'
                elif label == 'company':
                    tag[j[0]] = 'B-COMPANY'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-COMPANY'
                elif label == 'game':
                    tag[j[0]] = 'B-GAME'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-GAME'
                elif label == 'government':
                    tag[j[0]] = 'B-GOVERNMENT'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-GOVERNMENT'
                elif label == 'movie':
                    tag[j[0]] = 'B-MOVIE'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-MOVIE'
                elif label == 'name':
                    tag[j[0]] = 'B-NAME'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-NAME'
                elif label == 'organization':
                    tag[j[0]] = 'B-ORGANIZATION'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-ORGANIZATION'
                elif label == 'position':
                    tag[j[0]] = 'B-POSITION'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-POSITION'
                elif label == 'scene':
                    tag[j[0]] = 'B-SCENE'
                    for k in range(j[0] + 1,j[1] + 1):
                        tag[k] = 'I-SCENE'
    dic['text'] = text
    dic['tag'] = tag
    dataset.append(dic)

labels = ['O', 'B-ADDRESS', 'I-ADDRESS', 'B-BOOK', 'I-BOOK', 'B-COMPANY', 'I-COMPANY',
        'B-GAME', 'I-GAME', 'B-GOVERNMENT', 'I-GOVERNMENT', 'B-MOVIE', 'I-MOVIE', 
        'B-NAME', 'I-NAME', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-POSITION',
        'I-POSITION', 'B-SCENE', 'I-SCENE']
# initialize mat
pi_mat = {}
transform_mat = {}
observe_mat = [defaultdict(int) for _ in range(len(labels))]
state_cnt = [0 for _ in range(len(labels))]
for label in labels:
    pi_mat[label] = 0
    transform_mat[label] = np.zeros(len(labels))
    for newlabel in range(len(labels)):
        transform_mat[label][newlabel] = 0
# count
for item in dataset:
    pi_mat[item['tag'][0]] += 1
    for i in range(len(item['tag']) - 1):
        transform_mat[item['tag'][i]][labels.index(item['tag'][i + 1])] += 1
    for i in range(len(item['tag'])):
        state_cnt[labels.index(item['tag'][i])] += 1
        observe_mat[labels.index(item['tag'][i])][item['text'][i]] += 1
# normalize

pi_mat = np.array(list(pi_mat.values()))
pi_mat = pi_mat / pi_mat.sum()

transform_mat = np.array([transform_mat[l] for l in labels])
transform_mat = transform_mat / transform_mat.sum(axis=1, keepdims=True)

for i in range(len(labels)):
    for k in observe_mat[i].keys():
        observe_mat[i][k] /= state_cnt[i]

# viterbi
# read valid
true_example_cnt = defaultdict(int)
example_cnt = defaultdict(int)
prediction_cnt = defaultdict(int)

pred_list = []
tag_list = []
with tqdm(total=len(dev)) as pbar:
    for sentence in dev:

        new_dic = {}
        new_text = sentence['text']
        tag = ['O' for _ in range(len(new_text))]

        pred = viterbi(sentence['text'], pi_mat, transform_mat, observe_mat, labels)
        
        for label in sentence['label']:
            for i in sentence['label'][label]:
                for j in sentence['label'][label][i]:
                    if label == 'address':
                        tag[j[0]] = 'B-ADDRESS'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-ADDRESS'
                    elif label == 'book':
                        tag[j[0]] = 'B-BOOK'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-BOOK'
                    elif label == 'company':
                        tag[j[0]] = 'B-COMPANY'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-COMPANY'
                    elif label == 'game':
                        tag[j[0]] = 'B-GAME'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-GAME'
                    elif label == 'government':
                        tag[j[0]] = 'B-GOVERNMENT'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-GOVERNMENT'
                    elif label == 'movie':
                        tag[j[0]] = 'B-MOVIE'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-MOVIE'
                    elif label == 'name':
                        tag[j[0]] = 'B-NAME'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-NAME'
                    elif label == 'organization':
                        tag[j[0]] = 'B-ORGANIZATION'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-ORGANIZATION'
                    elif label == 'position':
                        tag[j[0]] = 'B-POSITION'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-POSITION'
                    elif label == 'scene':
                        tag[j[0]] = 'B-SCENE'
                        for k in range(j[0] + 1,j[1] + 1):
                            tag[k] = 'I-SCENE'
        # print(tag)
        # print(pred)
        pre = []
        for i in range(len(sentence['text'])):
            pretag = labels[int(pred[i])]
            pre.append(pretag)
        pred_list.append(pre)
        tag_list.append(tag)
        pbar.update(1)

print(classification_report(tag_list, pred_list))