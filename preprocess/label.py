# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:02:44 2018

@author: caobo
"""

import pandas as pd
import numpy as np
import math
import os
import tushare as ts
import datetime
import calendar

os.chdir("D:\AIfinal")


df = pd.read_csv('cleaned_2.0.csv', index_col = 0, encoding = 'gb18030')
d = ts.get_stock_basics()
def toutf(x):
    return x.encode("utf-8")
def touni(x):
    return unicode(x, "utf-8") 
d[u'name2'] = d['name'].apply(touni)
def getcode(x):
    try:
        return str(d[d[u'name2'] == x].index)[9:15]
    except:
        return 0
df['code'] = df[u'公司名称'].apply(getcode)
df['date'] = df[u'研报发布日期'].apply(toutf)

df = df.reset_index(drop = True)


def getNextMonday(date):

    today = datetime.datetime.strptime(date,'%Y-%m-%d')

    #print today

    oneday = datetime.timedelta(days = 1)

    m1 = calendar.MONDAY

    while today.weekday() != m1:
        today += oneday

    nextMonday = today.strftime('%Y-%m-%d')

    return nextMonday

def getAb1(codes,date):
    try:
        
        startdate = getNextMonday(date)
        enddate = (datetime.datetime.strptime(startdate,'%Y-%m-%d') + datetime.timedelta(days = 1) * 91).strftime('%Y-%m-%d')
        #print (startdate)
        #print (enddate)
    
        if codes[0:3] == '000':
            market = '399106'
        elif codes[0:3] == '002':
            market = '399005'
        elif codes[0:3] == '300':
            market = '399006'
        else:
            market = '000001'
        
        kdata = ts.get_k_data(code = codes, start = startdate, end = enddate, ktype = 'w')
        mdata = ts.get_k_data(code = market, start = startdate, end = enddate, ktype = 'w', index = True)
        kdata=kdata.reset_index(drop = True)
        mdata=mdata.reset_index(drop = True)
        #print (kdata)
        #print (mdata)
    
    
        ret_w = (kdata['close'][0] - kdata['open'][0]) / (kdata['open'][0]) - (mdata['close'][0] - mdata['open'][0]) / (mdata['open'][0])
        a = kdata.shape[0]-1
        b = mdata.shape[0]-1
        ret_s = (kdata['close'][a] - kdata['open'][0]) / (kdata['open'][0]) - (mdata['close'][b] - mdata['open'][0]) / (mdata['open'][0])
    except:
        ret_w = 'nan'
        ret_s = 'nan'
    return (ret_w,ret_s)

df['ret_w'] = df['label1'].apply(lambda x: float(x))
df['ret_s'] = df['ret_w']


for i in range(df.shape[0]):
    (df['ret_w'][i], df['ret_s'][i]) = getAb1(df['code'][i], df['date'][i])
    if i % 10 == 0 :
        print('【' + str(i) + '】')

dfp = df.drop_duplicates(inplace = False)

def label_w(r):
    if r > 0.035:
        return 5
    elif r > 0.01:
        return 4
    elif r >= -0.01:
        return 3
    elif r >= -0.035:
        return 2
    else:
        return 1
    

def label_s(r):
    if r > 0.122:
        return 5
    elif r > 0.03:
        return 4
    elif r >= -0.03:
        return 3
    elif r >= -0.122:
        return 2
    else:
        return 1

dfp['label_week'] = dfp['ret_w'].apply(label_w)     
dfp['label_season'] = dfp['ret_s'].apply(label_s)    

dfp.to_csv('cleaned_4.0.csv',encoding ='gb18030')