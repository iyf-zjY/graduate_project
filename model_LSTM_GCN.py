import torch.nn as nn
import torch
import torch.nn.functional as F
from pygcn.pygcn.layers import GraphConvolution
from utils import *


class LSTM_GCN(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size,
                 seq_len, feature_dimension,gcn_hidden_size, nclass, dropout=0.0,
                 lstm_num_layers=1, lstm_bias=True, batch_first=True,
                 lstm_bidirectional=False):
        super(LSTM_GCN, self).__init__()
        self.lstm_input_size = lstm_input_size  # feature amount per day (per stock)
        self.lstm_hidden_size = lstm_hidden_size
        self.seq_len = seq_len
        self.feature_dimension = feature_dimension
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
        #self.fc = nn.Linear(in_features=self.seq_len*self.lstm_hidden_size, out_features=self.feature_dimension)
        self.gc1 = GraphConvolution(self.lstm_hidden_size, self.gcn_hidden_size)
        self.gc2 = GraphConvolution(self.gcn_hidden_size, self.nclass)

    def _get_features(self, x):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :return:
        '''
        assert self.batch_first
        assert stock_amount == x.shape[0]
        lstm_out, (h_n, _) = self.LSTM(x)
        lstm_out = lstm_out.reshape((stock_amount, -1))
        h_n = h_n.reshape((stock_amount, -1))

        return h_n

    def forward(self, x, adj):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :param adj: adj matrix
        :return: loss
        '''
        feats = self._get_features(x)
        # feats = self.fc(feats)
        gc1_out = self.gc1(feats, adj)
        gc1_out = F.dropout(gc1_out, self.dropout, training=self.training)
        gc2_out = self.gc2(gc1_out, adj)
        return F.softmax(gc2_out, dim=1)


class ContrastLSTM(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size,
                 seq_len, feature_dimension,gcn_hidden_size, nclass, dropout=0.0,
                 lstm_num_layers=1, lstm_bias=True, batch_first=True,
                 lstm_bidirectional=False):
        super(ContrastLSTM, self).__init__()
        self.lstm_input_size = lstm_input_size  # feature amount per day (per stock)
        self.lstm_hidden_size = lstm_hidden_size
        self.seq_len = seq_len
        self.feature_dimension = feature_dimension
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
        self.fc = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.nclass)
        #self.gc1 = GraphConvolution(self.lstm_hidden_size, self.gcn_hidden_size)
        #self.gc2 = GraphConvolution(self.gcn_hidden_size, self.nclass)

    def _get_features(self, x):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :return:
        '''
        assert self.batch_first
        assert stock_amount == x.shape[0]
        lstm_out, (h_n, _) = self.LSTM(x)
        lstm_out = lstm_out.reshape((stock_amount, -1))
        h_n = h_n.reshape((stock_amount, -1))

        return h_n

    def forward(self, x, adj):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :param adj: adj matrix
        :return: loss
        '''
        feats = self._get_features(x)
        fcout = self.fc(feats)
        # gc1_out = self.gc1(feats, adj)
        # gc1_out = F.dropout(gc1_out, self.dropout, training=self.training)
        # gc2_out = self.gc2(gc1_out, adj)
        return F.softmax(fcout, dim=1)


class ContrastGCN(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size,
                 seq_len, feature_dimension,gcn_hidden_size, nclass, dropout=0.0,
                 lstm_num_layers=1, lstm_bias=True, batch_first=True,
                 lstm_bidirectional=False):
        super(ContrastGCN, self).__init__()
        self.lstm_input_size = lstm_input_size  # feature amount per day (per stock)
        self.lstm_hidden_size = lstm_hidden_size
        self.seq_len = seq_len
        self.feature_dimension = feature_dimension
        self.gcn_hidden_size = gcn_hidden_size
        self.nclass = nclass
        self.dropout = dropout
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bias = lstm_bias
        self.batch_first = batch_first
        self.lstm_bidirectional = lstm_bidirectional
        # self.LSTM = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
        #                     num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
        #                     batch_first=self.batch_first)
        self.fc = nn.Linear(in_features=self.seq_len*self.lstm_input_size, out_features=self.lstm_hidden_size)
        self.gc1 = GraphConvolution(self.lstm_hidden_size, self.gcn_hidden_size)
        self.gc2 = GraphConvolution(self.gcn_hidden_size, self.nclass)

    def _get_features(self, x):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :return:
        '''
        assert self.batch_first
        assert stock_amount == x.shape[0]
        # lstm_out, (h_n, _) = self.LSTM(x)
        # lstm_out = lstm_out.reshape((stock_amount, -1))
        # h_n = h_n.reshape((stock_amount, -1))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def forward(self, x, adj):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :param adj: adj matrix
        :return: loss
        '''
        feats = self._get_features(x)
        # feats = self.fc(feats)
        gc1_out = self.gc1(feats, adj)
        gc1_out = F.dropout(gc1_out, self.dropout, training=self.training)
        gc2_out = self.gc2(gc1_out, adj)
        return F.softmax(gc2_out, dim=1)


class LSTM_GCN_Attention(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size,
                 seq_len, feature_dimension, gcn_hidden_size, nclass, node_count, dropout=0.0,
                 lstm_num_layers=1, lstm_bias=True, batch_first=True,
                 lstm_bidirectional=False):
        super(LSTM_GCN_Attention, self).__init__()
        self.lstm_input_size = lstm_input_size  # feature amount per day (per stock)
        self.lstm_hidden_size = lstm_hidden_size
        self.node_count = node_count
        self.seq_len = seq_len
        self.feature_dimension = feature_dimension
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
        #self.fc = nn.Linear(in_features=self.seq_len*self.lstm_hidden_size, out_features=self.feature_dimension)
        self.gc1 = GraphConvolution(self.lstm_hidden_size, self.gcn_hidden_size)
        self.gc2 = GraphConvolution(self.gcn_hidden_size, self.nclass)
        self.attn_weight = nn.Parameter(torch.Tensor(1, self.node_count))

    def _get_features(self, x):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :return:
        '''
        assert self.batch_first
        assert stock_amount == x.shape[0]
        lstm_out, (h_n, _) = self.LSTM(x)
        lstm_out = lstm_out.reshape((stock_amount, -1))
        h_n = h_n.reshape((stock_amount, -1))

        return h_n

    def forward(self, x, adj):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :param adj: adj matrix
        :return: loss
        '''
        feats = self._get_features(x)
        # feats = self.fc(feats)
        gc1_out = self.gc1(feats, adj)
        gc1_out = F.dropout(gc1_out, self.dropout, training=self.training)
        gc2_out = self.gc2(gc1_out, adj)
        return torch.matmul(F.softmax(self.attn_weight, dim=1), F.softmax(gc2_out, dim=1)) #1*2
        # return F.softmax(gc2_out, dim=1)


class ContrastGCN_Attention(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size,
                 seq_len, feature_dimension,gcn_hidden_size, nclass, node_count, dropout=0.0,
                 lstm_num_layers=1, lstm_bias=True, batch_first=True,
                 lstm_bidirectional=False):
        super(ContrastGCN_Attention, self).__init__()
        self.lstm_input_size = lstm_input_size  # feature amount per day (per stock)
        self.lstm_hidden_size = lstm_hidden_size
        self.seq_len = seq_len
        self.node_count = node_count
        self.feature_dimension = feature_dimension
        self.gcn_hidden_size = gcn_hidden_size
        self.nclass = nclass
        self.dropout = dropout
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bias = lstm_bias
        self.batch_first = batch_first
        self.lstm_bidirectional = lstm_bidirectional
        # self.LSTM = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
        #                     num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
        #                     batch_first=self.batch_first)
        self.fc = nn.Linear(in_features=self.seq_len*self.lstm_input_size, out_features=self.lstm_hidden_size)
        self.gc1 = GraphConvolution(self.lstm_hidden_size, self.gcn_hidden_size)
        self.gc2 = GraphConvolution(self.gcn_hidden_size, self.nclass)
        self.attn_weight = nn.Parameter(torch.Tensor(1, self.node_count))

    def _get_features(self, x):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :return:
        '''
        assert self.batch_first
        assert stock_amount == x.shape[0]
        # lstm_out, (h_n, _) = self.LSTM(x)
        # lstm_out = lstm_out.reshape((stock_amount, -1))
        # h_n = h_n.reshape((stock_amount, -1))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def forward(self, x, adj):
        '''
        :param x: 3D tensor of shape (batch, seq_len, lstm_input_size)
        :param adj: adj matrix
        :return: loss
        '''
        feats = self._get_features(x)
        # feats = self.fc(feats)
        gc1_out = self.gc1(feats, adj)
        gc1_out = F.dropout(gc1_out, self.dropout, training=self.training)
        gc2_out = self.gc2(gc1_out, adj)
        return torch.matmul(F.softmax(self.attn_weight, dim=1), F.softmax(gc2_out, dim=1))  # 1*2
        # return F.softmax(gc2_out, dim=1)