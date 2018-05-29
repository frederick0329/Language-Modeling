import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class RNNLM(nn.Module):

    def __init__(self, vocabulary_size, embedding_size, hidden_units, num_layers):
        super(RNNLM, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        dropout = 0.2
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.RNN = nn.LSTM(embedding_size, hidden_units, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_units, vocabulary_size)
        self.init_weights()
      
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.embedding(x))
        out, hidden = self.RNN(emb, hidden)
        out = self.drop(out)
        batch_size = out.size(0)
        out = self.linear(out.contiguous().view(-1, self.hidden_units))
        return out.view(batch_size, -1, self.vocabulary_size), hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_units), 
                torch.zeros(self.num_layers, batch_size, self.hidden_units))
