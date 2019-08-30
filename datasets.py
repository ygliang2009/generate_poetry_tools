#!/usr/bin/python
import os
import json
import numpy as np
import re

datasets_dir = 'data_raw'
t = 'tang'
author = None
maxlen = 125

para_words_t = []

def sentence_parse(para):
    result, number = re.subn(u"（.*）", "", para)
    result, number = re.subn(u"{.*}", "", result)
    result, number = re.subn(u"《.*》", "", result)
    result, number = re.subn(u"《.*》", "", result)
    result, number = re.subn(u"[\]\[]", "", result)
    r = ""
    for s in result:
        if s not in set('0123456789-'):
            r += s
    r, number = re.subn(u"。。", u"。", r)
    return r


def padding_sequence(para_data_t, maxlen, padding_val):
    para_padding_t = []
    para_len_t = [len(d) for d in para_data_t]
    for d in para_data_t:
        if len(d) >= maxlen:
            para_padding_t.append(d[0:maxlen])
        else:
            t = (maxlen - len(d)) * [padding_val]
            t.extend(d)
            para_padding_t.append(t)

    return para_padding_t


def get_data():
    for data_file in os.listdir(datasets_dir):
        if data_file[0:5] == 'poet.' and data_file[5:9] == t:
            with open(os.path.join(datasets_dir, data_file), 'r') as dsf:
                poetry_dataset_lists = json.loads(dsf.read())
                for _, poetry_dataset in enumerate(poetry_dataset_lists):
                    if 'author' in poetry_dataset:
                        if author is not None and poetry_dataset['author'] != author:
                            continue
                        if 'paragraphs' not in poetry_dataset:
                            continue
                        paragraphs = poetry_dataset['paragraphs']
                        pdata = ""
                        for sentences in paragraphs:
                            sentences = sentences.replace('，',',').replace('。',',').replace('：','').replace(' ', '')
                            pdata += sentences   
                        pdata = sentence_parse(pdata) 
                        para_words_t.append(pdata)         
    words_set = {words for sentences in para_words_t for words in sentences}
    idx2word = {idx:word for idx, word in enumerate(words_set)}
    #space 
    idx2word[len(idx2word)] = '</s>'
    word2idx = {word:idx for idx, word in idx2word.items()}
    para_data_t = [[word2idx[word] for word in sentences] for sentences in para_words_t]
    
    para_padding_t = padding_sequence(para_data_t, maxlen, word2idx['</s>'])
    para_padding_a = np.asarray(para_padding_t)
    np.savez_compressed('aaaaa.npz', data = para_padding_a, wd2idx = word2idx, idx2wd = idx2word) 
    return para_padding_a, word2idx, idx2word


if __name__ == '__main__':
    para_padding_t, _, _ = get_data()
    print (para_padding_t)
