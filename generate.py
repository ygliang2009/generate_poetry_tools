#!/usr/bin/python
import numpy as np
import nets 
import torch
import os

prefix_words = u'床前明月光'
cuda_available = torch.cuda.is_available()
path = 'models'
gen_max_length = 200
datasets = np.load('aaaaa.npz')
word2idx, idx2word = datasets['wd2idx'].item(), datasets['idx2wd'].item()

net = nets.PoetryModel(256, len(word2idx))
checkpoint = torch.load(os.path.join(path,'best_model.t7'))
net.load_state_dict(checkpoint['net'])

if cuda_available is True:
    net.cuda()

def generate(prefix_words):
    results = []
    results_var = ""
    prefix_words_list = list(prefix_words)
    prefix_codec_list = [word2idx[word] for word in prefix_words_list]
    hidden = None
    output = None

    for i in range(gen_max_length):
        if i < len(prefix_codec_list):
            word = prefix_codec_list[i]
            word_t = torch.from_numpy(np.asarray([word]))
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = idx2word[top_index]
            results.append(w)
            results_var += w
            word_t = torch.from_numpy(np.asarray([top_index]))

        if cuda_available is True:
            word_t = word_t.view(1, 1)
            word_t = word_t.cuda()

        output, hidden = net(word_t, hidden)
    return results,results_var

if __name__ == '__main__':
    results, results_var = generate(prefix_words)
    print (results)
    print (results_var)
