import time
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
import os,sys

from model_LSTM_GCN import LSTM_GCN
from utils import *
from pygcn.pygcn.utils import accuracy
from tqdm import tqdm


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=45, help='Random seed.')
    parser.add_argument('-opt', '--optimizer', default='SGD')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=15,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    # parser.add_argument('--training_percent', type=float, default=0.7,
    #                     help='percentage of data used for training')
    # parser.add_argument('--valid_percent', type=float, default=0.15,
    #                     help='percentage of data used for validation')
    # parser.add_argument('--test_percent', type=float, default=0.15,
    #                     help='percentage of data used for test')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features_list, labels_list, \
    features_list_test, labels_list_test = load_pickle(
        ['adj', 'features_list', 'labels_list', 'test_features_list', 'test_labels_list'],
        'data/pickle_seq20_pwin3/'
    )
    train_val_index = int(len(features_list) * 0.8)
    # print(train_val_index)
    # input()
    # features_list_val, labels_list_val = \
    #     deepcopy(features_list[: train_val_index)]), \
    #     deepcopy(features_list[: train_val_index)])
    features_list_val = deepcopy(features_list[train_val_index:])
    labels_list_val = deepcopy(labels_list[train_val_index:])
    features_list = features_list[:train_val_index]
    labels_list = labels_list[:train_val_index]

    # features_list: list of 3D Tensor of shape (stock_amount, seq_len, lstm_input_size=3)
    #labels_list: list of 1D Tensor of shape (stock_amount)
    assert args.hidden < features_list[0].shape[1]
    model = LSTM_GCN(
        lstm_input_size=3,
        lstm_hidden_size=1,
        seq_len=features_list[0].shape[1],
        gcn_hidden_size=args.hidden,
        nclass=2,
        dropout=args.dropout)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = None
        raise ValueError(args.optimizer)

    if args.cuda:
        model.cuda()
        # features = features.cuda()
        features_list = [t.cuda() for t in features_list]
        labels_list = [t.cuda() for t in labels_list]
        features_list_test = [t.cuda() for t in features_list_test]
        labels_list_test = [t.cuda() for t in labels_list_test]
        features_list_val = [t.cuda() for t in features_list_val]
        labels_list_val = [t.cuda() for t in labels_list_val]
        adj = adj.cuda()
        # labels = labels.cuda()

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        loss_train_list = []
        acc_train_list = []
        print("Epoch: {}, begin training...".format(epoch+1))
        for feat, label in zip(features_list, labels_list):
            model.zero_grad()
            output = model(feat, adj)
            loss_train = F.cross_entropy(output, label)
            loss_train_list.append(loss_train.item())
            # By default, the losses are averaged over each loss element in the batch.
            acc_train = accuracy(output, label)
            acc_train_list.append(acc_train.item())
            loss_train.backward()
            optimizer.step()
            # validation
        model.eval()
        loss_v = []
        acc_v = []
        print("Epoch: {}, begin validating...".format(epoch+1))
        for feat_v, label_v in zip(features_list_val, labels_list_val):
            output_v = model(feat_v, adj)
            loss_v.append(F.cross_entropy(output_v, label_v).item())
            acc_v.append(accuracy(output_v, label_v).item())
        loss_vm = np.array(loss_v).mean()
        acc_vm = np.array(acc_v).mean()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(np.array(loss_train_list).mean()),
              'acc_train: {:.4f}'.format(np.array(acc_train_list).mean()),
              'loss_val: {:.4f}'.format(loss_vm),
              'acc_val: {:.4f}'.format(acc_vm),
              'time: {:.4f}s'.format(time.time() - t))
    print("Optimization Finished!")

    # begin test
    model.eval()
    loss_test_list = []
    acc_test_list = []
    for feat_t, label_t in tqdm(zip(features_list_test, labels_list_test)):
        output_t = model(feat_t, adj)
        loss_test_list.append(F.cross_entropy(output_t, label_t).item())
        acc_test_list.append(accuracy(output_t, label_t).item())
    print("Test set results:",
          "loss= {:.4f}".format(np.array(loss_test_list).mean()),
          "accuracy= {:.4f}".format(np.array(acc_test_list).mean()))

