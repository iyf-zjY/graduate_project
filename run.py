import time
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import os,sys

from model_LSTM_GCN import *
from TPA_LSTM import *
from utils import *
from pygcn.pygcn.utils import accuracy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, matthews_corrcoef,f1_score
from datetime import datetime
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # Training settings

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--save', type=bool, default=True,
                        help='Save the model during training')
    parser.add_argument('--seed', type=int, default=45, help='Random seed.')
    parser.add_argument('-opt', '--optimizer', default='Adam')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=40,
                        help='Number of hidden units of gcn.')
    parser.add_argument('--lstm_hidden_size', type=int, default=50,
                       help='lstm hidden size.')
    parser.add_argument('--feature_dimension', type=int, default=60,
                        help='the feature dimension which is the input of GCN')   # no use
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--clip', type=float, default=15.0,
                        help='gradient clipping by norm')
    parser.add_argument('--dynamic_lr', type=bool, default=False,
                        help='tuning learning rate dynamically')
    parser.add_argument('--only_test', type=bool, default=False)
    parser.add_argument('--train_saved_model', type=bool, default=False)  #train到一半接着train
    parser.add_argument('--model', type=str, default='GCN_HS300')
    parser.add_argument('--save_path', type=str, default='saved_model/GCN_WWW_pwin14_HS300_0925/')
    # parser.add_argument('--training_percent', type=float, default=0.7,
    #                     help='percentage of data used for training')
    # parser.add_argument('--valid_percent', type=float, default=0.15,
    #                     help='percentage of data used for validation')
    # parser.add_argument('--test_percent', type=float, default=0.15,
    #                     help='percentage of data used for test')
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features_list, labels_list, \
    features_list_test, labels_list_test = load_pickle(
        ['adj', 'features_list', 'labels_list', 'test_features_list', 'test_labels_list'],
        'data/pickle_seq30_pwin14_intv1_GCN_forWWW_HS300/'
    )
    train_val_index = int(len(features_list) * 1.0)
    # print(train_val_index)
    # input()
    # features_list_val, labels_list_val = \
    #     deepcopy(features_list[: train_val_index)]), \
    #     deepcopy(features_list[: train_val_index)])
    features_list_val = deepcopy(features_list[train_val_index:])
    labels_list_val = deepcopy(labels_list[train_val_index:])
    features_list = features_list[:train_val_index]
    labels_list = labels_list[:train_val_index]
        # features_list: list of 3D Tensor of shape (stock_amount, seq_len, lstm_input_size=15)
        #labels_list: list of 1D Tensor of shape (stock_amount)
    if not args.only_test:
        model = LSTM_GCN(
            lstm_input_size=5,
            lstm_hidden_size=args.lstm_hidden_size,
            seq_len=features_list[0].shape[1],
            feature_dimension=args.feature_dimension,
            gcn_hidden_size=args.hidden,
            nclass=2,
            dropout=args.dropout)
        if args.model == 'GCN_HS300':  # hs300 dedicate
            model = LSTM_GCN_Attention(
                lstm_input_size=5,
                lstm_hidden_size=args.lstm_hidden_size,
                seq_len=features_list[0].shape[1],
                feature_dimension=args.feature_dimension,
                gcn_hidden_size=args.hidden,
                nclass=2,
                node_count=features_list[0].shape[0],
                dropout=args.dropout
            )
        elif args.model == 'ContrastGCN_HS300':
            model = ContrastGCN_Attention(
                lstm_input_size=5,
                lstm_hidden_size=args.lstm_hidden_size,
                seq_len=features_list[0].shape[1],
                feature_dimension=args.feature_dimension,
                gcn_hidden_size=args.hidden,
                nclass=2,
                node_count=features_list[0].shape[0],
                dropout=args.dropout
            )

        elif args.model == 'LSTM':
            model = ContrastLSTM(
                lstm_input_size=5,
                lstm_hidden_size=args.lstm_hidden_size,
                seq_len=features_list[0].shape[1],
                feature_dimension=args.feature_dimension,
                gcn_hidden_size=args.hidden,
                nclass=2,
                dropout=args.dropout)
        elif args.model == 'ContrastGCN':
            model = ContrastGCN(
                lstm_input_size=5,
                lstm_hidden_size=args.lstm_hidden_size,
                seq_len=features_list[0].shape[1],
                feature_dimension=args.feature_dimension,
                gcn_hidden_size=args.hidden,
                nclass=2,
                dropout=args.dropout)
        elif args.model == 'TPA_LSTM':
            args_1 = arg_parse_TPA()
            # 记得不同的特征窗口长度需要手动改里边的window参数
            model = TPA_LSTM_modified(args_1, 5, hs300=False)

        elif args.model == 'TPA_LSTM_HS300':
            args_1 = arg_parse_TPA()
            # 记得不同的特征窗口长度需要手动改里边的window参数
            model = TPA_LSTM_modified(args_1, 5, hs300=True, node_count=features_list[0].shape[0])

        if args.train_saved_model:
            model = torch.load('saved_model/LSTM_H=30/Final_2020-04-12 140035.947072.pkl')

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
            # labels = labels.cuda()
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
        train_acc = []
        val_acc = []

        # trick一下，看看test集合最高能达到多少acc
        best_acc_test = 0.0
        best_test_model = None

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
                clip_grad_norm_(model.parameters(), args.clip)
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
            #在test集合上也看看
            loss_t = []
            acc_t = []
            test_predict = None
            test_label = None
            for feat_t, label_t in zip(features_list_test, labels_list_test):
                output_t = model(feat_t, adj)
                loss_t.append(F.cross_entropy(output_t, label_t).item())
                acc_t.append(accuracy(output_t, label_t).item())
                test_predict = output_t if test_predict is None else torch.cat((test_predict, output_t), dim=0)
                test_label = label_t if test_label is None else torch.cat((test_label, label_t), dim=0)

            acc_tm = np.array(acc_t).mean()
            # try:
            #     tp, fp, tn, fn, f1 = compute_F1(test_predict, test_label)
            # except:
            #     continue
            x_values, y_values, auc = compute_auc(test_predict[:, 1], test_label)
            mcc = compute_mcc(test_predict, test_label)
            print('Epoch: {:04d}'.format(epoch + 1),
                  'acc_test: {:.4f}'.format(acc_tm),
                  # "F1 score= {:.4f}".format(f1),
                  "AUC= {:.4f}".format(auc),
                  "MCC= {:.4f}".format(mcc),
                  "Mac-F1 = {:.4f}".format(f1_score(test_label.tolist(), test_predict.max(1)[1].tolist(), average='macro'))
                  )
            if best_acc_test < acc_tm:
                best_acc_test = acc_tm
                # best_test_model = deepcopy(model)

            train_acc.append(np.array(acc_train_list).mean())
            val_acc.append(acc_vm)
            if args.save and epoch % 10 == 0 and epoch != 0:
                    torch.save(model, args.save_path + 'epoch{}_'.format(epoch) + '_pickle_seq60_pwin14_intv1_{}forWWW_'.format(args.model)
                           + str(datetime.now().date()) + '.pkl')
        print("Optimization Finished!")
        if best_test_model is not None:
            torch.save(best_test_model, args.save_path + 'best_test_model_pickle_seq60_pwin14_intv1_{}forWWW_'.format(args.model)
                               + str(datetime.now().date()) + '.pkl')

        # begin test
        model.eval()
        loss_test_list = []
        acc_test_list = []

        test_predict = None
        test_label = None

        for feat_t, label_t in tqdm(zip(features_list_test, labels_list_test)):
            output_t = model(feat_t, adj)
            test_predict = output_t if test_predict is None else torch.cat((test_predict,output_t), dim=0)
            test_label = label_t if test_label is None else torch.cat((test_label, label_t), dim=0)
            loss_test_list.append(F.cross_entropy(output_t, label_t).item())

            acc_test_list.append(accuracy(output_t, label_t).item())

        tp, fp, tn, fn, f1 = compute_F1(test_predict, test_label)
        x_values, y_values, auc = compute_auc(test_predict[:, 1], test_label)
        mcc = compute_mcc(test_predict, test_label)
        print("Test set results:",
              "loss= {:.4f}".format(np.array(loss_test_list).mean()),
              "accuracy= {:.4f}".format(np.array(acc_test_list).mean()),
              "TP= {}, FP= {}, TN= {}, FN={}".format(tp,fp,tn,fn),
              "F1 score= {:.4f}".format(f1),
              "AUC= {:.4f}".format(auc),
              "MCC= {:.4f}".format(mcc))
        # if args.save:
        #     torch.save(model, 'saved_model\\Final_TPA_LSTM' + str(datetime.now()).replace(':', '') + '.pkl')

        plt.plot(range(args.epochs), train_acc, color='red')
        plt.plot(range(args.epochs), val_acc, color='green')
        plt.show()
        plt.figure()
        plt.plot(x_values, y_values)
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.show()

    else:
        model = torch.load('F:\python_project\graduate_project\saved_model\没聚类特征对比试验_lstmh=30_gcnh=10\\best_test_model_2020-04-19 141413.777046.pkl')
        model.eval()
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
        loss_test_list = []
        acc_test_list = []

        test_predict = None
        test_label = None

        for feat_t, label_t in tqdm(zip(features_list_test, labels_list_test)):
            output_t = model(feat_t, adj)
            test_predict = output_t if test_predict is None else torch.cat((test_predict,output_t), dim=0)
            test_label = label_t if test_label is None else torch.cat((test_label, label_t), dim=0)
            loss_test_list.append(F.cross_entropy(output_t, label_t).item())
            acc_test_list.append(accuracy(output_t, label_t).item())

        tp, fp, tn, fn, f1 = compute_F1(test_predict, test_label)
        x_values, y_values, auc = compute_auc(test_predict[:, 1], test_label)
        mcc = compute_mcc(test_predict, test_label)

        print("Test set results:",
              "loss= {:.4f}".format(np.array(loss_test_list).mean()),
              "accuracy= {:.4f}".format(np.array(acc_test_list).mean()),
              "TP= {}, FP= {}, TN= {}, FN={}".format(tp,fp,tn,fn),
              "F1 score= {:.4f}".format(f1),
              "AUC= {:.4f}".format(auc),
              "MCC = {:.4f}".format(mcc))
        # if args.save:
        #     torch.save(model, 'saved_model\\Final_' + str(datetime.now()).replace(':', '') + '.pkl')
        plt.figure()
        plt.plot(x_values, y_values)
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.show()