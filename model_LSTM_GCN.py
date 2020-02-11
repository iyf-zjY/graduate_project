import torch.nn as nn
import torch.nn.functional as F
from pygcn.pygcn.layers import GraphConvolution
from utils import *


class LSTM_GCN(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size,
                 seq_len, gcn_hidden_size, nclass, dropout=0.0,
                 lstm_num_layers=1, lstm_bias=True, batch_first=True,
                 lstm_bidirectional=False):
        super(LSTM_GCN, self).__init__()
        self.lstm_input_size = lstm_input_size  # feature amount per day (per stock)
        self.lstm_hidden_size = lstm_hidden_size  # should be 1
        self.seq_len = seq_len
        self.gcn_hidden_size = gcn_hidden_size
        self.nclass = nclass
        self.dropout = dropout
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bias = lstm_bias
        self.batch_first = batch_first
        self.lstm_bidirectional = lstm_bidirectional
        self.LSTM = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
                            batch_first=self.batch_first)
        self.gc1 = GraphConvolution(self.seq_len, self.gcn_hidden_size)
        self.gc2 = GraphConvolution(self.gcn_hidden_size, self.nclass)

    def _get_features(self, x):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :return: features: 2D tensor of shape(lstm_hidden_size, seq_len)
        '''
        assert self.batch_first
        assert stock_amount == x.shape[0]
        lstm_out, _ = self.LSTM(x)
        lstm_out = lstm_out.view(stock_amount, -1)
        return lstm_out

    def forward(self, x, adj):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :param adj: adj matrix
        :return: loss
        '''
        feats = self._get_features(x)
        gc1_out = self.gc1(feats, adj)
        gc1_out = F.dropout(gc1_out, self.dropout, training=self.training)
        gc2_out = self.gc2(gc1_out, adj)
        return F.softmax(gc2_out, dim=1)
