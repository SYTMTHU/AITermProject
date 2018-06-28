import numpy as np

import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor


class Classifier(torch.nn.Module):
    def __init__(self, nentities, nout, nhid1= 128, nhid2 = 256):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(nentities+1, nhid1)
        self.linear2 = torch.nn.Linear(nhid1, nhid2)
        self.linear3 = torch.nn.Linear(nhid2, nout)
      #  self.sm = torch.nn.Softmax()
        self.init_weights()
        self.nentities = nentities
        self.nout = nout
        self.nhid1 = nhid1
        self.nhid2 = nhid2

    def forward(self, x):
        if type(x) == list:
            x = Variable(Tensor(x))
        h_relu = self.linear1(x).clamp(min = 0)
        h_relu_2 = self.linear2(h_relu).clamp(min = 0)
        y_pred = self.linear3(h_relu_2)
     #   y_output = self.sm(y_pred)
        return y_pred

    def init_weights(self):
        initrange = 0.1
        self.linear1.bias.data.fill_(0)
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.fill_(0)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.fill_(0)
        self.linear3.weight.data.uniform_(-initrange, initrange)