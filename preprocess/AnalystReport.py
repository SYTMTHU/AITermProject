
# coding: utf-8

# In[1]:


import thulac 
import snownlp
from bs4 import BeautifulSoup
import re
import numpy as np 
import pandas as pd 
import os
os.chdir("/Users/terrenceshi/Desktop/report17-firm/firmabstr/")


# In[2]:


path = "/Users/terrenceshi/Desktop/report17-firm/firmabstr"
reports=os.listdir(path)
pd_report17=pd.DataFrame(np.zeros((len(reports),len(range(100)))), index=set(reports))


# In[3]:


for report in reports:
    try:
        file=open(path+"/"+report)
        soup = BeautifulSoup(file, "lxml")
        pd_report17.loc[report, 0]=soup.h3.text
        pd_report17.loc[report, 1]=soup.select("[class~=txt_02] script")[0].get_text().split('"',2)[1]
        pd_report17.loc[report, 2]=soup.select("[class~=text_01]  span")[0].get_text()
        pd_report17.loc[report, 3]=soup.find_all(text=re.compile("hyname"))[0].split("'",2)[1]
        pd_report17.loc[report, 4]=soup.h3.text.split(" ")[1]
        pd_report17.loc[report, 5]=soup.select("[class~=txt_02]")[1].get_text().split(" ")[-1:]
        for part in range(2,len(soup.select("[class~=txt_02]"))):
            pd_report17.loc[report, part+4]=soup.select("[class~=txt_02]")[part].get_text()
    except:
        continue 


# In[4]:


pd_report17


# In[5]:


pd_report17.to_excel("/Users/terrenceshi/Desktop/report17-firm/abstract17.xlsx")


# In[71]:





# In[72]:





# In[73]:




