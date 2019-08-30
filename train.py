#!/usr/bin/python
import torch
import torch.nn as nn
import nets
import datasets as ds
from torch.utils.data import DataLoader
import os
import numpy as np
import sys
import tqdm

resume = False

if len(sys.argv) > 1 and sys.argv[1] == 'r':
    resume = True

hidden_number = 256
learning_rate = 0.01 
bs = 128
saving_path = 'models'

start_epoch = 0
end_epoch = 200

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

cuda_available = torch.cuda.is_available() 
#cuda_available = False
para_padding_a, word2ix, ix2word = ds.get_data()
para_padding_t = torch.from_numpy(para_padding_a)

net = nets.PoetryModel(hidden_number, len(word2ix))

if resume is True:
    checkpoints = torch.load(os.path.join(saving_path, 'best_model.t7'))
    net.load_state_dict(checkpoints['net'])
    start_epoch = checkpoints['epoch']

net = net.cuda() if cuda_available is True else net

dataloader = DataLoader(para_padding_t, batch_size = bs, shuffle = True, num_workers = 1)

optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

min_loss_t = 10000 
print ('datasets length = %d, batch_size = %d, start_epoch = %d'%(len(dataloader), bs, start_epoch)) 
for epoch_idx in range(start_epoch, end_epoch):
    loss_buff_a = []
    for batch_idx, data in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        data = data.long().transpose(1, 0).contiguous()
        inputs = data[0:-1,:]
        target = data[1:,:]

        if cuda_available is True:
            inputs = inputs.cuda() 
            target = target.cuda()
 
        target = target.view(-1)
        output, _ = net(inputs)
        loss = criterion(output, target)
        loss_buff_a.append(loss.item())
        loss.backward()
        optimizer.step()

    loss_mean_val = np.mean(loss_buff_a)
    if loss_mean_val < min_loss_t:
        print ('saving ...')
        min_loss_t = loss_mean_val
        state = {
            'net':net.state_dict() if cuda_available else net,
            'epoch':epoch_idx,
            'min_loss': min_loss_t
        }
        torch.save(state, os.path.join(saving_path, 'best_model.t7'))

    print ('epoch = %d, loss_mean_val = %f, best_loss = %f'%(epoch_idx, loss_mean_val, min_loss_t))
