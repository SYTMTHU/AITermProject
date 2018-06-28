# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:40:22 2018

@author: caobo
"""

import pandas as pd
import numpy as np
import math
import os

os.chdir("D:\AIfinal")


dfcp = pd.read_csv('clean_fea.csv', index_col = 0, encoding = 'gb18030')
dfn = pd.read_csv('cleaned_2.0.csv', index_col = 0, encoding = 'gb18030')
dfwn = dfcp[dfcp['chi'] > 2.5]
dfwn = dfwn.sort_values(by="chi" , ascending=False).reset_index(drop = True)


def evalu(s):
    if (u'买入评级' == s or u'推荐评级' == s or u'强烈推荐评级' == s):
        return 3
    elif (u'增持评级' == s or u'谨慎推荐评级' == s or u'谨慎买入评级' == s):
        return 2
    elif (u'中性评级' == s or u'持有评级' == s or u'减持评级' == s):
        return 1
    else:
        return 0
    
#dfn[u'统一评级'] = dfn[u'推荐评级'].apply(evalu)
#dfn = dfn[dfn[u'统一评级'] > 0]


label = u'统一评级'
def chi(t,c,N,df):
    dftemp1 = df[df[label] == c]
    dftemp2 = df[df[label] != c]
    dftemp1['ind'] = dftemp1['words'].apply(lambda x: x.count(t))
    dftemp2['ind'] = dftemp2['words'].apply(lambda x: x.count(t))
    A = float(dftemp1[dftemp1['ind'] >= 1].shape[0])
    C = float(dftemp1[dftemp1['ind'] == 0].shape[0])
    B = float(dftemp2[dftemp2['ind'] >= 1].shape[0])
    D = float(dftemp2[dftemp2['ind'] == 0].shape[0])
    #print(A)
    #print(B)
    #print(C)
    #print(D)
    
    try :
        fi = dftemp1['ind'].sum() / (A+C)
        ci = A/(A+B)
        di = A/(A+C)
        if (A*D - B*C) <= 0:
            Chi_2 = 0
        else:
            Chi_2 = ((N*(A*D - B*C)*(A*D - B*C)) / ((A+B)*(A+C)*(D+B)*(D+C)))*((fi+ci+di)/3)
    except:
        print(t)
        Chi_2 = 0.0
    return Chi_2

feature_chi_1 = []
feature_chi_2 = []
feature_chi_3 = []
feature_chi = []
feature_label = []
for i in range(dfwn.shape[0]):
    C1 = chi(dfwn['feature'][i], 1, dfn.shape[0], dfn)
    C2 = chi(dfwn['feature'][i], 2, dfn.shape[0], dfn)
    C3 = chi(dfwn['feature'][i], 3, dfn.shape[0], dfn)
    feature_chi_1.append(C1)
    feature_chi_2.append(C2)
    feature_chi_3.append(C3)
    feature_chi.append(max(C1,C2,C3))
    if max(C1,C2,C3) == C1:
        a = 1
    elif max(C1,C2,C3) == C2:
        a = 2
    else:
        a = 3
    feature_label.append(a)
    if i % 50 == 0 :
        print('【' + str(i) + '】')

try:
    dfwn['chi_2_l1'] = feature_chi_1
    dfwn['chi_2_l2'] = feature_chi_2
    dfwn['chi_2_l3'] = feature_chi_3
    dfwn['chi_2'] = feature_chi
    dfwn['label_2'] = feature_label
    dfwn.to_csv('clean_fea_pj2.csv',encoding = 'gb18030')
except:
    print('error')