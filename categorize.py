import pandas as pd
pd.set_option('max_colwidth', 100000)
import numpy as np
import re
from sys import exit
from copy import deepcopy

f_categories = './categories'
f = open(f_categories, 'r')

df_parted_text = pd.read_excel('cleaned_4.xlsx')
df_raw_text_17 = pd.read_excel('finish17.xlsx')
df_raw_text_1316 = pd.read_excel('finish1316.xlsx')
df_word2chinlabel = pd.read_excel('word2chi.xlsx')


def setpattern(f_categories):
    f = f_categories
    categories = f.readlines()
    cat2patt = dict()
    patterns = []
    for line in categories:
        line = line.strip().split()
        cname = line[0]
        cpattern = line[1]
        if len(line) > 1:
            for word in line[2:]:
                cpattern += '|' + word
        cpattern = '(' + cpattern + ')'
        patterns.append(cpattern)
        cat2patt[cname] = cpattern
    f.seek(0)
    return cpattern, patterns

def checkEntity(patterns, paraphrase):
    containEntity = [0] * (len(patterns) +1)
    for i in range(len(patterns)):
        pattern = patterns[i]
        result = re.search(pattern, paraphrase)
        if not result is None:
            containEntity[i+1] = 1
    if not (1 in containEntity):
        containEntity[0] = 1
    return  containEntity

def transWord2Chi(df_word2chinlabel):
    '''
    :param df_word2chinlabel:
    :return: dictionary of word2index
    '''
    word2index = dict()
    index2word = []
    index2chi = []
    word2chi = dict()
    for row in df_word2chinlabel.itertuples():
        label = int(row.label_2) - 2
        chi = label*float(row.chi_2)
        word = str(row.feature)
        word2index[word] = len(index2word)
        index2word.append(word)
        index2chi.append(chi)
        word2chi[word] = chi
    return word2index, index2word, index2chi, word2chi

def getEmotion(row_i, word2chi):
    emo = 0
    for word in row_i:
        if word in word2chi:
            emo += word2chi[word]
  #          print('emotion of {}'.format(word))
    return emo

def preprocess(df_raw_text, df_parted_text, f_categories, df_word2chinlabel, noutput=5, stockprices=None):
    count = 0
    cpattern , patterns = setpattern(f_categories)
    word2index, index2word, index2chi, word2chi = transWord2Chi(df_word2chinlabel)
    inpput = []
    target = []
    for row in df_parted_text.itertuples():
        sum_emo = np.zeros(len(patterns)+1)
        cname = row.companyname
        date = str(row.date.date())
        broker = row.broker
        raw_text = df_raw_text[(df_raw_text.companyname == cname) & (df_raw_text.date == date) & (df_raw_text.broker == broker)]
        if len(raw_text) > 1:
            print('more than one report has the same company name, date and broker as {}, {} and {}'.format(cname, date, broker))
            raw_text = pd.core.frame.DataFrame(raw_text.iloc[0]).transpose()
     #   elif len(raw_text) < 1:
      #      print('no report has company name, date and broker as {}, {} and {}'.format(cname, date, broker))
        for i in range(5, 26):
            raw_text_i = raw_text['index'+str(i)].to_string(index=False)
            raw_text_i = raw_text_i[2:-2]
            containEntity = checkEntity(patterns, raw_text_i)
      #      last_entity = deepcopy(containEntity) # TODO: not entity tag in a certain paragraph
            row_i = row.__getattribute__('index'+str(i))[1:-1].split(',')
            if len(row_i) > 0:
                emo = getEmotion(row_i, word2chi)
            else:
                emo = 0
            sum_emo += np.array(containEntity)*emo
        inpput.append(sum_emo)
      #  output_category = [0]*noutput
        output_category = int(row.__getattribute__('label_week')) - 1
        target.append(output_category)
    return inpput, target


if __name__ == '__main__':
    import pickle
    inpput, target = preprocess(df_raw_text_1316, df_parted_text, f, df_word2chinlabel)
    inpput_17, target_17 = preprocess(df_raw_text_17, df_parted_text, f, df_word2chinlabel)
    assert(len(inpput) == len(target))
    assert(len(inpput_17) == len(target_17))
    inpput.extend(inpput_17)
    target.extend(target_17)
    f_out_input = open('1316input', 'wb')
    f_out_target = open('1316target', 'wb')

    pickle.dump(inpput, f_out_input)
    pickle.dump(target, f_out_target)
    f_out_input.close()
    f_out_target.close()




