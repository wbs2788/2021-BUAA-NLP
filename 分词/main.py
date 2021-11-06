'''
Author: wbs2788
Date: 2021-11-04 10:10:15
LastEditTime: 2021-11-04 20:14:45
LastEditors: wbs2788
Description: Chinese Fenci
'''

import re
from tqdm import tqdm

f1 = open('C:\\Users\\surafce book2\\Desktop\\实践二：中文分词\\词典\\pku_training_words.utf8','r',encoding='utf-8')
f1 = f1.read()
dic = f1.split("\n")
maxlen = 0
dic = frozenset([d.strip() for d in dic])
for i in dic:
    maxlen = maxlen if maxlen > len(i) else len(i)

f1 = open('C:\\Users\\surafce book2\\Desktop\\实践二：中文分词\\停用词\\stop_word.txt','r',encoding='utf-8')
f1 = f1.read()
stp = f1.split("\n")
stp = frozenset([d.strip() for d in dic])

f1 = open('C:\\Users\\surafce book2\\Desktop\\实践二：中文分词\\待分词文件\\corpus.txt','r',encoding='utf-8')
f1 = f1.read()
text = f1.split("\n")

f1 = open('C:\\Users\\surafce book2\\Desktop\\实践二：中文分词\\分词对比文件\\gold.txt','r',encoding='utf-8')
f1 = f1.read()
ans = f1.split("\n")

def FMM(text:str) -> list:
    ans = []
    len_text = len(text)
    i = 0
    while i < len_text:
        end = i + maxlen
        if end >= len_text:
            end = len_text
        sub_text = text[i : end]
        for j in reversed(range(1, end - i)):
            if j == 1:
                ans.append((i, i + 1))
                break
            cur = sub_text[0 : j]
            if cur in dic:
                ans.append((i, i + j))
                i += len(cur) - 1
                break
        i += 1
    return ans

def BMM(text:str) -> list:
    ans = []
    len_text = len(text)
    i = len_text
    while i > 0:
        start = i - maxlen
        if start < 0:
            start = 0
        sub_text = text[start : i]
        len_sub = len(sub_text)
        for j in range(len_sub):
            if j == len_sub - 1:
                ans.append((i - 1, i))
                break
            cur = sub_text[j : len_sub]
            if cur in dic:
                ans.append((j + start, i))
                i -= len(cur) - 1
                break
        i -= 1
    return ans[::-1]

def BM(text:str, FMM:str, BMM:str) -> list:
    if len(FMM) != len(BMM):
        return FMM if len(FMM) < len(BMM) else BMM
    if FMM == BMM:
        return FMM
    cnt1, cnt2 = 0, 0
    for i in FMM:
        if len(i) == 1:
            cnt1 += 1
    for i in BMM:
        if len(i) == 1:
            cnt2 += 1
    return FMM if cnt1 < cnt2 else BMM

def shortestPath(text:str) -> list: # Floyd Warshall
    l = len(text)
    ans = []
    f = [1e10 for __ in range(0, l + 1 )]
    next = [i + 1 for i in range(0, l)]
    f[0] = 0
    
    for i in range(0, l):
        for j in range(i + 1, l + 1):
            dist = 1e10
            if j - i == 1:
                dist = 1
            if text[i:j] in dic:
                dist = f[i] + 1
            if dist < f[j]:
                f[j] = dist
                next[i] = j
    k = 0
    while k != l:
        ans.append((k, next[k]))
        k = next[k]
    return ans

def list2str(l:list, t:str) -> list:
    ans = []
    for i in l:
        ans.append(t[i[0]:i[1]])
    return ans
        
def str2list(t:str) -> list:
    ans = []
    t = t.split('  ')
    cnt = 0
    for i in t:
        length = len(i)
        ans.append((cnt, cnt + length))
        cnt += length
    return ans

def estimatePR(pred:list, target:list) -> int:
    ans = 0
    for cur in target:
        if cur in pred:
            ans += 1
    return ans

if __name__ == '__main__':
    bm_cnt = 0
    sp_cnt = 0
    key_cnt = 0
    pre1 = 0
    pre2 = 0
    with tqdm(total=len(text)) as pbar:
        for i, a in zip(text, ans):
            fmm = FMM(i)
            bmm = BMM(i)
            ans1 = BM(i, fmm, bmm)
            bm_cnt += len(ans1)
            ans2 = shortestPath(i)
            sp_cnt += len(ans2)
            key = str2list(a)
            key_cnt += len(key)
            pre1 += estimatePR(ans1, key)
            pre2 += estimatePR(ans2, key)
            pbar.update(1)
        
    print(float(pre1)/bm_cnt, float(pre1)/key_cnt)
    print(float(pre2)/sp_cnt, float(pre2)/key_cnt)