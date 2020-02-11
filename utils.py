import numpy as np
import sklearn
import csv
import sys,os
import pickle
import scipy.sparse as sp
from Kmeans.Kmeans_cDTW_DBA import DBA_iteration
from pygcn.pygcn.utils import *
stock_amount = 296
tol_parameter = 0.3
cluster_data_name = '180101-190530'
cluster_ans_dir = 'data/price_data/' + cluster_data_name + '/train_result/'
cluster_cent_file = cluster_data_name + '_12_cent.csv'
cluster_mem_file = cluster_data_name + '_12_mem.csv'
stock_idx_file = 'stock_idx_' + cluster_data_name + '.txt'
up_and_down_file = 'up_and_down_' + cluster_data_name + '.csv'
z_scored_file = 'z_scored_' + cluster_data_name + '.csv'

train_data_name = '180101-190530'
train_dir = 'data/price_data/' + train_data_name + '/'
train_z_scored_file = 'z_scored_' + train_data_name + '.csv'
train_stock_idx_file = 'stock_idx_' + train_data_name + '.txt'
train_hs300_file = 'hs300/z_scored_' + train_data_name + '.csv'

test_data_name = '190601-190930'
test_dir = 'data/price_data/' + test_data_name + '/'
test_z_scored_file = 'z_scored_' + test_data_name + '.csv'
test_stock_idx_file = 'stock_idx_' + test_data_name + '.txt'
test_hs300_file = 'hs300/z_scored_' + test_data_name + '.csv'

#制作样本的原始CSV格式同聚类用到的z_scored
#转为content格式的，是已经z-scored的


def init_info():
    mem_dir = os.path.join(cluster_ans_dir, cluster_mem_file)
    data_ud_dir = os.path.join(cluster_ans_dir, up_and_down_file)
    data_dir = os.path.join(cluster_ans_dir, z_scored_file)
    stock_idx_cluster = []
    mem_cluster = []
    with open(mem_dir, 'r') as f:
        lines = csv.reader(f)
        for i, line in enumerate(lines):
            stock_idx_cluster.append(line[0].replace('s',''))
            mem_cluster.append(int(line[1]))
    with open(data_ud_dir, 'r') as f:
        data_ud_cluster = list(csv.reader(f))
    data_ud_cluster = np.array(data_ud_cluster, dtype=float)
    data_ud_cluster = np.array(data_ud_cluster, dtype=int)

    with open(os.path.join(train_dir, train_z_scored_file), 'r') as f:
        train_z_scored_data = list(csv.reader(f))
    with open(os.path.join(test_dir, test_z_scored_file), 'r') as f:
        test_z_scored_data = list(csv.reader(f))

    with open(os.path.join(train_dir, train_stock_idx_file), 'r') as f:
        train_idx = list(csv.reader(f))
    with open(os.path.join(test_dir, test_stock_idx_file), 'r') as f:
        test_idx = list(csv.reader(f))
    assert len(train_z_scored_data) == len(train_idx)
    assert len(test_z_scored_data) == len(test_idx)

    train_z_scored_data_dict = dict()
    test_z_scored_data_dict = dict()
    for stock_no, single_data in zip(train_idx, train_z_scored_data):
        train_z_scored_data_dict[stock_no[0].replace('s','')] = np.array(single_data, dtype=np.float32)
    for stock_no, single_data in zip(test_idx, test_z_scored_data):
        test_z_scored_data_dict[stock_no[0].replace('s','')] = np.array(single_data,dtype=np.float32)

    train_z_scored_data = np.array(train_z_scored_data, dtype=np.float32)
    test_z_scored_data = np.array(test_z_scored_data, dtype=np.float32)
    assert train_z_scored_data.shape[0] == stock_amount  # hs300 - 4
    train_idx = [t[0].replace('s', '') for t in train_idx]
    test_idx = [t[0].replace('s', '') for t in test_idx]

    with open(os.path.join(train_dir, train_hs300_file), 'r') as f:
        train_hs300_z_scored_data = np.array(list(csv.reader(f)), dtype=np.float32).flatten()
    with open(os.path.join(test_dir, test_hs300_file), 'r') as f:
        test_hs300_z_scored_data = np.array(list(csv.reader(f)),  dtype=np.float32).flatten()

    return mem_cluster, data_ud_cluster, stock_idx_cluster, \
           train_z_scored_data_dict, train_z_scored_data, train_idx, train_hs300_z_scored_data,\
           test_z_scored_data_dict, test_z_scored_data, test_idx, test_hs300_z_scored_data


def ask_shape_similar(tol_param, stock_no, mem, data_ud, stock_idx):
    if stock_no not in stock_idx:
        return [stock_no]
    this_type = mem[stock_idx.index(stock_no)]
    data_ud_this_type = []
    idx_this_type = []
    for j in range(len(mem)):
        if mem[j] == this_type:
            data_ud_this_type.append(data_ud[j])
            idx_this_type.append(stock_idx[j])
    majority_direction = data_ud[stock_idx.index(stock_no)]

    obvious_stock_no = []
    for seq, stock_ud in enumerate(data_ud_this_type):
        diff_day = 0
        for day in range(data_ud.shape[1]):
            if majority_direction[day] != 0:
                if stock_ud[day] != majority_direction[day]:
                    diff_day += 1
        # print("diff day: {} stock: {},".format(diff_day,idx_this_type[seq]))
        # print("diff rate: {}".format(float(diff_day) / float(Data_ud.shape[1])))
        # input()
        if float(diff_day) / float(data_ud.shape[1]) <= tol_param:
            obvious_stock_no.append(idx_this_type[seq])

    return obvious_stock_no


def generate_edges_from_ssn(ssn_file):
    # from node2vec
    ssn_name = os.path.basename(ssn_file).replace('.csv','.edges')
    with open(ssn_file, 'r') as f:
        write_edges = []
        reader = csv.reader(f)
        for edge in reader:
            edge = edge[0].split(' ')
            write_edges.append(edge[0]+' '+edge[1]+'\n')
    with open(ssn_name,'w') as f:
        f.writelines(write_edges)


def generate_cluster_feats(target_stock_no, all_z_scored_data_dict,
                           begin_index, end_index,
                           mem_cluster, data_ud_cluster, stock_idx_cluster,
                           ways='DBA'):
    """
    get relative stocks from the result of cluster
    and generate fusion feature of the target stock
    :param target_stock_no: type:str
    :param all_z_scored_data_dict: type: dict of stockno-z_scored_data(整个时间区间的data, 用以制作样本)
    :param begin_index:
    :param end_index:  [begin_index, end_index)
    :param mem_cluster:
    :param data_ud_cluster:
    :param stock_idx_cluster:
    :param ways:
    :return:
    """
    assert ways in ['DBA', 'numerical_mean']
    similar_stocks = ask_shape_similar(tol_parameter, target_stock_no,
                                       mem_cluster, data_ud_cluster, stock_idx_cluster)
    similar_stocks = [all_z_scored_data_dict[t][begin_index:end_index] for t in similar_stocks]
    if ways == 'DBA':
        fusion_feature = DBA_iteration(all_z_scored_data_dict[target_stock_no][begin_index:end_index], similar_stocks)
    else:
        fusion_feature = [
            [seq[t] for seq in similar_stocks]
            for t in range(len(all_z_scored_data_dict[target_stock_no][begin_index: end_index]))
        ]
        fusion_feature = [
            np.array(t).mean() for t in fusion_feature
        ]
    return fusion_feature


def generate_hs300_feats(begin_index, end_index,
                         hs300_data):
    return hs300_data[begin_index:end_index]


def get_target_label(window_data):
    """
    :param window_data: 2D ndarray : (stock_amount, window_len_+1)
    :return: target label 1 if up 0 else
    """
    labels = []
    for i in range(window_data.shape[0]):
        if list(window_data[i])[-1] > list(window_data[i])[0]:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def generate_samples(features, seq_len_features, predict_window, gen_sample_interval,
                     idx, z_scored_data_dict, mem_cluster, data_ud_cluster, stock_idx_cluster,
                     hs300_data):
    features_list, labels_list = [], []
    i = 0
    while i < features.shape[1]:
        if i + seq_len_features + predict_window - 1 >= features.shape[1]:
            break
        sample_i = []
        for j in range(features.shape[0]):
            fusion_cluster_feature = generate_cluster_feats(idx[j], z_scored_data_dict,
                                                            i, i + seq_len_features,
                                                            mem_cluster, data_ud_cluster,
                                                            stock_idx_cluster, 'numerical_mean')
            hs300_feats = generate_hs300_feats(i, i + seq_len_features, hs300_data)
            tmp_data = [features[j][i:i + seq_len_features],
                        fusion_cluster_feature,
                        hs300_feats]
            sample_i.append(
                [[tmp_data[0][j], tmp_data[1][j], tmp_data[2][j]] for j in range(len(tmp_data[0]))]
            )
        labels_i = get_target_label(features[:, i + seq_len_features - 1:i + seq_len_features + predict_window])
        features_list.append(torch.FloatTensor(np.array(sample_i)))
        labels_list.append(torch.LongTensor(np.array(labels_i)))
        i += gen_sample_interval
    return features_list, labels_list


def load_data_and_gen_samples(dataset="180101-190930",
                              weighted_graph=True, weighted_graph_file='data/ssn/ssn_line.csv',
                              seq_len_features=20, predict_window=3, gen_sample_interval=3):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    mem_cluster, data_ud_cluster, stock_idx_cluster, \
    z_scored_data_dict, features, idx, train_hs300, \
    test_z_scored_data_dict, test_features, test_idx, test_hs300 = init_info()

    # features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    for i in range(len(idx)):
        assert idx[i] == test_idx[i]
    # build graph
    idx_map = {j: i for i, j in enumerate(idx)}

    with open(weighted_graph_file, 'r') as f:
        reader = csv.reader(f)
        edges_unordered = []
        for edge_info in reader:
            edge_info = edge_info[0].split(' ')
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
            for weight_edge_info in weight_edges:
                i, j, w = weight_edge_info[0].split(' ')
                adj_dok[idx_map[i], idx_map[j]] = int(w)
        adj = adj_dok.tocoo()

    # build symmetric adjacency matrix
    #毕设项目中可以去掉这句话，并且从这个构造可以看出，并非必须0-1阵
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    assert features.shape[1] > gen_sample_interval
    features_list, labels_list = generate_samples(features, seq_len_features, predict_window,
                                                  gen_sample_interval, idx, z_scored_data_dict,
                                                  mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                  train_hs300)
    test_features_list, test_labels_list = generate_samples(test_features, seq_len_features, predict_window,
                                                            gen_sample_interval, test_idx, test_z_scored_data_dict,
                                                            mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                            test_hs300)
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    save_as_pickle([adj, features_list, labels_list, test_features_list, test_labels_list],
                   ['adj','features_list','labels_list','test_features_list','test_labels_list'],
                   'data/pickle_seq20_pwin3/')
    # return adj, features_list, labels_list, test_features_list, test_labels_list


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj: type sp.coo_matrix
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def save_as_pickle(data_list, name_list, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    assert len(data_list) == len(name_list)
    for i, data in enumerate(data_list):
        with open(directory + name_list[i] + '.pickle', 'wb') as f:
            pickle.dump(data, f)


def load_pickle(name_list, directory):
    data_list = []
    for file_name in name_list:
        with open(directory+file_name+'.pickle', 'rb') as f:
            data_list.append(pickle.load(f))
    return data_list


def count_category_balance(labels_list):
    total_amount = len(labels_list)*len(labels_list[0])
    count_p = 0
    for labels in labels_list:
        for t in labels:
            if t == 1:
                count_p += 1

    print(float(count_p) / float(total_amount))


if __name__ == '__main__':
    load_data_and_gen_samples()
    # adj, features_list, labels_list, \
    # features_list_test, labels_list_test = load_pickle(
    #     ['adj', 'features_list', 'labels_list', 'test_features_list', 'test_labels_list'],
    #     'data/pickle/'
    # )
    # print(adj.shape)
    # print(len(features_list))
    # print(len(features_list_test))
    # print(features_list[0].shape)
    # print(labels_list[0].shape)
    # print(features_list_test[0].shape)
    # print(labels_list_test[0].shape)
    # generate_edges_from_ssn('F:\python_project\graduate_project\data\ssn\\node2vec.csv')
    # adj = sp.coo_matrix(([1,2,3], ((0,1,2), (0,0,0))),
    #                     shape=(3, 3),
    #                     dtype=np.int32)
    # print(adj.todok()[2,0])