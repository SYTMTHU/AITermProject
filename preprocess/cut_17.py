# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:36:28 2018

@author: caobo
"""
import pandas as pd
import numpy as np
import codecs  
import os  
import shutil  
import jieba  
import jieba.analyse
import jieba.posseg as pseg

os.chdir("D:\AIfinal")

df = pd.read_csv('abstract17.csv', encoding = 'gb18030')
dfcpw = pd.read_csv('1316_corpus.csv', encoding = 'gb18030')
stopwords = {}.fromkeys([line.strip() for line in codecs.open('D:\AIfinal\stopwords.txt', encoding='UTF-8')]) #停用词表
stopflags = ['n','ns','nr','nt','nz','m','x','f']

#dic = {}
w = []
f = []
words = pseg.cut(df['6'][2])
for word, flag in words:
    #dic[word] = flag
    w.append(word)
    f.append(flag)
dic = {'word':w,'flag':f}
segdf = pd.DataFrame(dic)

def seg(sentence):
    seg_r = jieba.lcut(sentence)
    seg_list = []
    for seg in seg_r:         
        if seg not in stopwords:  
            if len(seg)>1:         
                seg_list.append(seg)
    return seg_list

def seg_f(sentence):
    seg_r = pseg.lcut(sentence)
    seg_list = []
    for seg,flag in seg_r:         
        if (seg not in stopwords and flag not in stopflags):  
            if len(seg)>1:         
                seg_list.append(seg)
    return seg_list    

df1 =df.copy()
df2 = df.copy()

for i in range(5,26):
    #df1[str(i)] = df[str(i)].apply(seg)
    df2[str(i)] = df[str(i)].apply(seg_f)
    print i
    
cp_word = list(dfcpw['feature'])   #语料库的词

def wordtocp(sentence):
    for i in range(len(sentence)):
        if sentence[i] not in cp_word:
            cp_word.append(sentence[i])

for i in range(5,26):
    #df1[str(i)] = df[str(i)].apply(seg)
    df2[str(i)].apply(wordtocp)
    print i
    


#df1.to_csv('1316.csv', encoding = 'gb18030')
df2.to_csv('17_f.csv', encoding = 'gb18030')

dfcp = pd.DataFrame({'feature':cp_word})
dfcp.to_csv('corpus.csv', encoding = 'gb18030')

        
    