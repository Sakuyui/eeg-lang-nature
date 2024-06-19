import torch
import torch.nn as nn

def EletrodeValueSequentialModel(nn.Module):
    def __init__(self, args):
        self.channels = args['cnt_channels']
        self.rnn = None
    
    def forward(self, x_t):
        return rnn(x_t)


