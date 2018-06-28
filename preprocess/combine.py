
# coding: utf-8

# In[1]:


import thulac 
import snownlp
from bs4 import BeautifulSoup
import re
import numpy as np 
import pandas as pd 
import os


# In[2]:


abstract=pd.read_excel("/Users/terrenceshi/Desktop/report17-firm/stock/TRD_Co.xlsx")


# In[3]:


report17=pd.read_excel("/Users/terrenceshi/Desktop/report17-firm/abstract17.xlsx", parse_dates=True)
marketret=pd.read_excel("/Users/terrenceshi/Desktop/report17-firm/stock/TRD_Dalym.xlsx", parse_dates=True)


# In[4]:


merged17=pd.merge(abstract, report17, left_on='Stknme',right_on='公司名称')


# In[5]:


stkret1=pd.read_excel("/Users/terrenceshi/Desktop/report17-firm/stock/TRD_Dalyr.xlsx", parse_dates=True)
stkret2=pd.read_excel("/Users/terrenceshi/Desktop/report17-firm/stock/TRD_Dalyr1.xlsx", parse_dates=True)


# In[28]:


stkret=pd.concat([stkret1,stkret2])


# In[29]:


stkret=pd.merge(stkret, marketret, on=['Markettype','Trddt'])


# In[30]:


stkret=stkret.sort_values(by=['Stkcd','Trddt'])
stkret['num']=range(len(stkret))


# In[31]:


stkret['abnormalew']=stkret['Dretwd']-stkret['Dretwdeq']
stkret['abnormalvw']=stkret['Dretwd']-stkret['Dretwdtl']


# In[32]:


stkret['rank']=stkret['num'].groupby(stkret['Stkcd']).rank(ascending=1)


# In[33]:


stkret


# In[34]:


for i in range(-2,4):
    stkret["rank"+str(i)]=stkret['rank']-i
    stkret["abnormalew"+str(i)]=stkret['abnormalew']
    stkret["abnormalvw"+str(i)]=stkret['abnormalvw']
    temp=stkret[['Stkcd',"rank"+str(i),"abnormalew"+str(i),"abnormalvw"+str(i)]]
    stkret_acc=stkret.drop(["abnormalew"+str(i),"abnormalvw"+str(i),"rank"+str(i)],axis=1)
    stkret=pd.merge(stkret_acc,temp,left_on=['Stkcd','rank'],right_on=['Stkcd',"rank"+str(i)],suffixes=['',''])


# In[36]:


pd_finish=pd.merge(stkret, merged17, left_on=['Stkcd','Trddt'], right_on=['Stkcd','研报发布日期'])


# In[37]:


pd_finish.to_excel("/Users/terrenceshi/Desktop/report17-firm/stock/finish17.xlsx")


# In[35]:


stkret


# In[21]:



stkret["rank1"]=stkret['rank']-1
stkret["abnormalew1"]=stkret['abnormalew']
stkret["abnormalvw1"]=stkret['abnormalvw']
   


# In[22]:


stkret


# In[23]:


temp=stkret[['Stkcd',"rank1","abnormalew1","abnormalvw1"]]
temp


# In[24]:


stkret_acc=stkret.drop(["abnormalew1","abnormalvw1","rank1"],axis=1)


# In[25]:


stkret_acc


# In[26]:


stkret_acc=pd.merge(stkret_acc,temp,left_on=['Stkcd','rank'],right_on=['Stkcd',"rank1"],suffixes=['',''])


# In[27]:


stkret_acc

