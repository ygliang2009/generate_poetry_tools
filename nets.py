import torch 
import torch.nn as nn

class PoetryModel(nn.Module):
    def __init__(self, hidden_dim, vocabulary_size):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocabulary_size, 128)
        self.lstm1 = nn.LSTM(128, self.hidden_dim, num_layers = 2)
        self.linear1 = nn.Linear(self.hidden_dim, vocabulary_size)

    def forward(self, x, hidden = None):
        seq_len, batch_size = x.shape
        if hidden is None:
            h_0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        x = self.embedding(x)
        x, hidden = self.lstm1(x, (h_0, c_0))
        x = x.view(seq_len * batch_size, -1)
        x = self.linear1(x)
        return x, hidden

if __name__ == '__main__':
    poetryModel = PoetryModel(256, 8100)
