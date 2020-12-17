import pickle
import scipy.sparse as sp
import numpy as np
import csv
from Kmeans.Kmeans_cDTW_DBA import DBA_iteration
from pygcn.pygcn.utils import *
from utils import generate_hs300_feats
import torch
import os


def init_info(train_dataset, test_dataset):
    attrs_list = ["close", "open", "high", "low", "volume"]
    train_data_name = train_dataset
    train_dir = 'data/price_data/' + train_data_name + '/'
    train_z_scored_file = '{}_z_scored_' + train_data_name + '.csv'
    train_stock_idx_file = 'close_stock_idx_' + train_data_name + '.txt'
    train_hs300_file = 'hs300/{}_z_scored_' + train_data_name + '.csv'

    test_data_name = test_dataset
    test_dir = 'data/price_data/' + test_data_name + '/'
    test_z_scored_file = '{}_z_scored_' + test_data_name + '.csv'
    test_stock_idx_file = 'close_stock_idx_' + test_data_name + '.txt'
    test_hs300_file = 'hs300/{}_z_scored_' + test_data_name + '.csv'

    all_features = []  # 5*295*len
    test_all_features = []
    for attr in attrs_list: #五个维度的特征都load进来
        with open(os.path.join(train_dir, train_z_scored_file.format(attr)), 'r') as f:
            all_features.append(list(csv.reader(f)))
        with open(os.path.join(test_dir, test_z_scored_file.format(attr)), 'r') as f:
            test_all_features.append(list(csv.reader(f)))

    with open(os.path.join(train_dir, train_stock_idx_file), 'r') as f:
        train_idx = list(csv.reader(f))
    with open(os.path.join(test_dir, test_stock_idx_file), 'r') as f:
        test_idx = list(csv.reader(f))
    assert len(all_features[0]) == len(train_idx)
    assert len(test_all_features[0]) == len(test_idx)

    # reshape一下 feature的shape: stockNum, days, 5
    train_z_scored_data = [
        [[t[i][j] for t in all_features]
         for j in range(len(all_features[0][0]))
         ]
        for i in range(len(all_features[0]))
    ]
    test_z_scored_data = [
        [[t[i][j] for t in test_all_features]
         for j in range(len(test_all_features[0][0]))
         ]
        for i in range(len(test_all_features[0]))
    ]
    train_z_scored_data_dict = dict()
    test_z_scored_data_dict = dict()
    for stock_no, single_data in zip(train_idx, train_z_scored_data):
        train_z_scored_data_dict[stock_no[0].replace('s', '')] = np.array(single_data, dtype=np.float32)
    for stock_no, single_data in zip(test_idx, test_z_scored_data):
        test_z_scored_data_dict[stock_no[0].replace('s', '')] = np.array(single_data, dtype=np.float32)

    train_z_scored_data = np.array(train_z_scored_data, dtype=np.float32)
    test_z_scored_data = np.array(test_z_scored_data, dtype=np.float32)
    # assert train_z_scored_data.shape[0] == stock_amount  # hs300 - 5
    train_idx = [t[0].replace('s', '') for t in train_idx]
    test_idx = [t[0].replace('s', '') for t in test_idx]

    all_hs300_features = []
    test_all_hs300_features = []
    for attr in attrs_list:
        with open(os.path.join(train_dir, train_hs300_file.format(attr)), 'r') as f:
            all_hs300_features.append(np.array(list(csv.reader(f)), dtype=np.float32).flatten())
        with open(os.path.join(test_dir, test_hs300_file.format(attr)), 'r') as f:
            test_all_hs300_features.append(np.array(list(csv.reader(f)), dtype=np.float32).flatten())

    train_hs300_z_scored_data = np.array([
        [t[i] for t in all_hs300_features]
        for i in range(len(all_hs300_features[0]))
    ], dtype=np.float32)
    test_hs300_z_scored_data = np.array([
        [t[i] for t in test_all_hs300_features]
        for i in range(len(test_all_hs300_features[0]))
    ], dtype=np.float32)
    return train_z_scored_data_dict, train_z_scored_data, train_idx, train_hs300_z_scored_data, \
           test_z_scored_data_dict, test_z_scored_data, test_idx, test_hs300_z_scored_data


def save_as_pickle(data_list, name_list, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    assert len(data_list) == len(name_list)
    for i, data in enumerate(data_list):
        with open(directory + name_list[i] + '.pickle', 'wb') as f:
            pickle.dump(data, f)


def get_target_label_hs300(hs300_data):
    if hs300_data[-1][0] > hs300_data[0][0]:
        return [1]
    else:
        return [0]


def gen_samples_DA_RNN(features, seq_len_features, predict_window, gen_sample_interval,
                       hs300_dedicate=False, hs300_data=None):
    features_list, labels_list = [], []
    i = 0
    while i < features.shape[1]:
        if i + seq_len_features + predict_window - 1 >= features.shape[1]:
            break
        if hs300_dedicate:
            tmp_data = [generate_hs300_feats(i,i + seq_len_features, hs300_data)]
            features_list.append(
                [np.array([tmp_data[0][s]]).flatten()
                 for s in range(len(tmp_data[0]))]
            )
            this_label = [t[0] for t in hs300_data[i:i + seq_len_features]]
            this_label.append(hs300_data[i + seq_len_features + predict_window - 1][0])
            labels_list.append(this_label)
        else:
            for j in range(features.shape[0]):
                tmp_data = [features[j][i:i + seq_len_features]]
                features_list.append(
                    [np.array([tmp_data[0][s]]).flatten()
                     for s in range(len(tmp_data[0]))]
                )
                #labels_list.append([t[0] for t in features[j][i:i+seq_len_features]])
                this_label = [t[0] for t in features[j][i:i+seq_len_features]]
                this_label.append(features[j][i + seq_len_features + predict_window - 1][0])
                labels_list.append(this_label)
        i += gen_sample_interval
    return features_list, labels_list


def generate_samples_SFM(features, file_name, multiVar=False):
    features_list = []
    save_path = "data/SFM_forWWW/" if not multiVar else "data/SFM_forWWW_multiVar/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(features.shape[0]):
        features_list.append(features[i,:,0] if not multiVar else features[i,:,:])
    features_list = np.array(features_list)
    np.save(save_path + file_name + ".npy", features_list)


def generate_samples_SFM_HS300(hs300_data, file_name, multiVar=False):
    features_list = []
    save_path = "data/SFM_forWWW_hs300/" if not multiVar else "data/SFM_forWWW_hs300_multiVar/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    features_list.append(hs300_data[:, 0] if not multiVar else hs300_data[:, :])
    features_list = np.array(features_list)
    np.save(save_path + file_name + ".npy", features_list)


def generate_samples_GCN(features, seq_len_features, predict_window, gen_sample_interval,
                         model='GCN', hs300_dedicate=False, hs300_data=None):
    features_list, labels_list = [], []
    i = 0
    while i < features.shape[1]:
        if i + seq_len_features + predict_window - 1 >= features.shape[1]:
            break
        sample_i = []
        for j in range(features.shape[0]):
            # seq_len_features * 5
            tmp_data = [features[j][i:i + seq_len_features]]
            sample_i.append(
                [np.array([tmp_data[0][s]]).flatten()
                 for s in range(len(tmp_data[0]))]
            )
            # 每个 sample: seq_len_features * 10
        from utils import get_target_label
        labels_i = get_target_label(features[:, i + seq_len_features - 1:i + seq_len_features + predict_window, 0]) \
            if not hs300_dedicate else get_target_label_hs300(hs300_data[i + seq_len_features - 1:i + seq_len_features+predict_window])
        if model == 'GCN' or model =="TPA_LSTM":
            features_list.append(torch.FloatTensor(np.array(sample_i)))
            labels_list.append(torch.LongTensor(np.array(labels_i)))
        elif model == 'InceptionTime':
            features_list += sample_i
            labels_list += labels_i
        else:
            raise ValueError("no this model {}".format(model))
        i += gen_sample_interval
    return features_list, labels_list


def generate_samples_InceptionTime_hs300(features, seq_len_features, predict_window, gen_sample_interval,
                         model='GCN', hs300_dedicate=False, hs300_data=None):
    features_list, labels_list = [], []
    i = 0
    while i < features.shape[1]:
        if i + seq_len_features + predict_window - 1 >= features.shape[1]:
            break
        tmp_data = [generate_hs300_feats(i, i + seq_len_features, hs300_data)]
        features_list.append(
            [np.array([tmp_data[0][s]]).flatten()
             for s in range(len(tmp_data[0]))]
        )
        # 每个 sample: seq_len_features * 10
        labels_i = get_target_label_hs300(
            hs300_data[i + seq_len_features - 1:i + seq_len_features + predict_window])
        labels_list += labels_i
        i += gen_sample_interval
    return features_list, labels_list


def load_data_and_gen_samples(train_dataset="170601-191129", test_dataset="191130-200530", seq_len_features=60,
                              predict_window=14, gen_sample_interval=1, model='normal',
                              weighted_graph=None, weighted_graph_file=None, hs300_dedicate=False):
    print('Loading {} {} dataset...'.format(train_dataset, test_dataset))
    z_scored_data_dict, features, idx, train_hs300, \
    test_z_scored_data_dict, test_features, test_idx, test_hs300 = init_info(train_dataset, test_dataset)
    for i in range(len(idx)):
        assert idx[i] == test_idx[i]

    if model == 'DA_RNN':
        feature_list, label_list = gen_samples_DA_RNN(features, seq_len_features,
                                                      predict_window, gen_sample_interval,
                                                      hs300_dedicate, train_hs300)
        test_feature_list, test_label_list = gen_samples_DA_RNN(test_features, seq_len_features,
                                                                predict_window, gen_sample_interval,
                                                                hs300_dedicate, test_hs300)
        save_as_pickle([feature_list,label_list,test_feature_list, test_label_list],
                       ['features_list', 'labels_list', 'test_features_list', 'test_labels_list'],
                       'data/pickle_seq{}_pwin{}_intv_{}_DARNN_ForWWW_hs300/'.format(seq_len_features,
                                                                               predict_window, gen_sample_interval))
    elif model == 'SFM':
        if not hs300_dedicate:
            generate_samples_SFM(features, "train_")
            generate_samples_SFM(test_features, "test_")
        else:
            generate_samples_SFM_HS300(train_hs300, "train_")
            generate_samples_SFM_HS300(test_hs300, "test_")

    elif model == 'SFM_multiVar':
        if not hs300_dedicate:
            generate_samples_SFM(features, "train_", multiVar=True)
            generate_samples_SFM(test_features, "test_", multiVar=True)
        else:
            generate_samples_SFM_HS300(train_hs300, "train_", multiVar=True)
            generate_samples_SFM_HS300(test_hs300, "test_", multiVar=True)
    elif model == 'GCN' or model == 'TPA_LSTM':
        assert weighted_graph is not None
        assert weighted_graph_file is not None
        idx_map = {j: i for i, j in enumerate(idx)}

        # 一些股票停盘的之类的，舆情有 但是没数据 删除这些股票
        ignore_edge_idx = []
        if 'pkl' not in weighted_graph_file:
            with open(weighted_graph_file, 'r') as f:
                reader = csv.reader(f)
                edges_unordered = []
                for edge_idx, edge_info in enumerate(reader):
                    edge_info = edge_info[0].split(' ')
                    if edge_info[0] not in idx_map.keys() or edge_info[1] not in idx_map.keys():
                        ignore_edge_idx.append(edge_idx)
                        continue
                    edges_unordered.append([edge_info[0], edge_info[1]])
            edges_unordered = np.array(edges_unordered, dtype=np.dtype(str))

            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                             dtype=np.int32).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(features.shape[0], features.shape[0]),
                                dtype=np.float32)
            if weighted_graph:
                with open(weighted_graph_file, 'r') as f:
                    weight_edges = csv.reader(f)
                    adj_dok = adj.todok()
                    for edge_idx, weight_edge_info in enumerate(weight_edges):
                        if edge_idx in ignore_edge_idx:
                            continue
                        i, j, w = weight_edge_info[0].split(' ')
                        adj_dok[idx_map[i], idx_map[j]] = int(w)
                adj = adj_dok.tocoo()
                from utils import normalize_adj
                adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        else:
            with open(weighted_graph_file, 'r') as f:
                adj = np.load(weighted_graph_file)
                print(adj.shape)
                input()
        features_list, labels_list = generate_samples_GCN(features, seq_len_features, predict_window,
                                                      gen_sample_interval,model, hs300_dedicate, train_hs300)
        test_features_list, test_labels_list = generate_samples_GCN(test_features, seq_len_features, predict_window,
                                                      gen_sample_interval,model, hs300_dedicate, test_hs300)
        from pygcn.pygcn.utils import sparse_mx_to_torch_sparse_tensor
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        save_as_pickle([adj, features_list, labels_list, test_features_list, test_labels_list],
                       ['adj', 'features_list', 'labels_list', 'test_features_list', 'test_labels_list'],
                       'data/pickle_seq{}_pwin{}_intv{}_GCN_forWWW_hs300/'.format(seq_len_features, predict_window,
                                                                        gen_sample_interval))

    elif model == 'InceptionTime':
        features_list, labels_list = generate_samples_GCN(features, seq_len_features, predict_window,
                                                         gen_sample_interval, model,hs300_dedicate, train_hs300) \
            if not hs300_dedicate else generate_samples_InceptionTime_hs300(
            features, seq_len_features, predict_window,
            gen_sample_interval, model, hs300_dedicate, train_hs300)
        test_features_list, test_labels_list = generate_samples_GCN(test_features, seq_len_features, predict_window,
                                                                    gen_sample_interval, model,hs300_dedicate, test_hs300) \
            if not hs300_dedicate else generate_samples_InceptionTime_hs300(
            features, seq_len_features, predict_window,
            gen_sample_interval, model, hs300_dedicate, train_hs300)
        features_list = np.array(features_list)
        labels_list = np.array(labels_list)
        test_features_list = np.array(test_features_list)
        test_labels_list = np.array(test_labels_list)
        print(features_list.shape, labels_list.shape, test_features_list.shape, test_labels_list.shape)
        save_path = 'data/InceptionTime_seq{}_pwin{}_intv{}_WWW_hs300/'.format(seq_len_features, predict_window,
                                                         gen_sample_interval)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + "X_TRAIN.npy", features_list)
        np.save(save_path + "Y_TRAIN.npy", labels_list)
        np.save(save_path + "X_TEST.npy", test_features_list)
        np.save(save_path + "Y_TEST.npy", test_labels_list)


if __name__ == '__main__':
    load_data_and_gen_samples(model='GCN',
                              weighted_graph=True, weighted_graph_file='data/ssn/fixed_ssn_line.csv',
                              hs300_dedicate=False)