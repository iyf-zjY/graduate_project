import numpy as np
import sklearn
import csv
import sys,os
import pickle
import scipy.sparse as sp
from Kmeans.Kmeans_cDTW_DBA import DBA_iteration
from pygcn.pygcn.utils import *
import torch
stock_amount = 295
tol_parameter = 0.3
cluster_data_name = '180101-190530'
cluster_ans_dir = 'data/price_data/' + cluster_data_name + '/train_result/'
cluster_cent_file = cluster_data_name + '_12_cent.csv'
cluster_mem_file = cluster_data_name + '_12_mem.csv'
stock_idx_file = 'stock_idx_' + cluster_data_name + '.txt'
up_and_down_file = 'up_and_down_' + cluster_data_name + '.csv'
z_scored_file = 'z_scored_' + cluster_data_name + '.csv'

train_data_name = '170601-191129'
train_dir = 'data/price_data/' + train_data_name + '/'
train_z_scored_file = '{}_z_scored_' + train_data_name + '.csv'
train_stock_idx_file = 'close_stock_idx_' + train_data_name + '.txt'
train_hs300_file = 'hs300/{}_z_scored_' + train_data_name + '.csv'

test_data_name = '191130-200530'
test_dir = 'data/price_data/' + test_data_name + '/'
test_z_scored_file = '{}_z_scored_' + test_data_name + '.csv'
test_stock_idx_file = 'close_stock_idx_' + test_data_name + '.txt'
test_hs300_file = 'hs300/{}_z_scored_' + test_data_name + '.csv'

#制作样本的原始CSV格式同聚类用到的z_scored
#转为content格式的，是已经z-scored的


def compute_F1(prediction, label):
    '''

    :param prediction:2d tensor
    :param label: tensor
    :return:
    '''
    preds = prediction.max(1)[1].type_as(label)
    TP = ((preds.data == 1) & (label.data == 1)).cpu().sum().item()
    FP = ((preds.data == 1) & (label.data == 0)).cpu().sum().item()
    TN = ((preds.data == 0) & (label.data == 0)).cpu().sum().item()
    FN = ((preds.data == 0) & (label.data == 1)).cpu().sum().item()
    print(TP, FP, TN, FN)
    p = TP / (TP+FP)
    r = TP / (TP+FN)
    return TP, FP, TN, FN, 2 * r * p / (r + p)


def compute_auc(prediction, label):
    '''
    :param prediction: 1d tensor
    :param label: tensor
    :return:
    '''
    zipped_list = list(zip(prediction, label))
    zipped_list = sorted(zipped_list, key=lambda x: x[0].item(), reverse=True)
    positive_count = label.sum().item()
    negative_count = len(zipped_list) - positive_count
    fpr = [0.0]
    tpr = [0.0]
    for tup in zipped_list:
        if tup[1].item() == 1:
            tpr.append(tpr[-1] + (1 / positive_count))
            fpr.append(fpr[-1])
        else:
            tpr.append(tpr[-1])
            fpr.append(fpr[-1] + (1/negative_count))
    auc = 0.0
    for i in range(len(tpr)-1):
        auc += (fpr[i+1]-fpr[i]) * (tpr[i+1]+tpr[i])
    auc /= 2
    return fpr, tpr, auc


def init_info():
    attrs_list = ["close", "open", "high", "low", "volume"]
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
        train_z_scored_data_dict[stock_no[0].replace('s','')] = np.array(single_data, dtype=np.float32)
    for stock_no, single_data in zip(test_idx, test_z_scored_data):
        test_z_scored_data_dict[stock_no[0].replace('s','')] = np.array(single_data,dtype=np.float32)

    train_z_scored_data = np.array(train_z_scored_data, dtype=np.float32)
    test_z_scored_data = np.array(test_z_scored_data, dtype=np.float32)

    assert train_z_scored_data.shape[0] == stock_amount  # hs300 - 5
    train_idx = [t[0].replace('s', '') for t in train_idx]
    test_idx = [t[0].replace('s', '') for t in test_idx]

    all_hs300_features = []
    test_all_hs300_features = []
    for attr in attrs_list:
        with open(os.path.join(train_dir, train_hs300_file.format(attr)), 'r') as f:
            all_hs300_features.append(np.array(list(csv.reader(f)), dtype=np.float32).flatten())
        with open(os.path.join(test_dir, test_hs300_file.format(attr)), 'r') as f:
            test_all_hs300_features.append(np.array(list(csv.reader(f)),  dtype=np.float32).flatten())

    train_hs300_z_scored_data = np.array([
        [t[i] for t in all_hs300_features]
        for i in range(len(all_hs300_features[0]))
    ],dtype=np.float32)
    test_hs300_z_scored_data = np.array([
        [t[i] for t in test_all_hs300_features]
        for i in range(len(test_all_hs300_features[0]))
    ],dtype=np.float32)

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
        # 20203.15,暂时别用，feature是多维度的，需要改一下DBA代码
        fusion_feature = DBA_iteration(all_z_scored_data_dict[target_stock_no][begin_index:end_index], similar_stocks)
    else:
        fusion_feature = [
            [seq[t] for seq in similar_stocks]
            for t in range(end_index - begin_index)
        ]

        fusion_feature = [
            [np.array([t[j][i] for j in range(len(t))]).mean() for i in range(len(t[0]))]
            for t in fusion_feature
        ]
    return fusion_feature


def generate_hs300_feats(begin_index, end_index,
                         hs300_data):
    return hs300_data[begin_index:end_index]


def get_target_label(window_data):
    """
    :param window_data: 3D ndarray : (stock_amount, window_len_+1,1)
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
                     hs300_data, model='normal'):
    features_list, labels_list = [], []
    i = 0
    while i < features.shape[1]:
        if i + seq_len_features + predict_window - 1 >= features.shape[1]:
            break
        sample_i = []
        for j in range(features.shape[0]):
            # seq_len_features * 5
            fusion_cluster_feature = generate_cluster_feats(idx[j], z_scored_data_dict,
                                                            i, i + seq_len_features,
                                                            mem_cluster, data_ud_cluster,
                                                            stock_idx_cluster, 'numerical_mean')
            hs300_feats = generate_hs300_feats(i, i + seq_len_features, hs300_data)

            tmp_data = [features[j][i:i + seq_len_features],
                        fusion_cluster_feature,
                        hs300_feats]
            sample_i.append(
                [np.array([tmp_data[0][s], tmp_data[1][s], tmp_data[2][s]]).flatten()
                 for s in range(len(tmp_data[0]))]
            )
            # 每个 sample: seq_len_features * 15
        labels_i = get_target_label(features[:, i + seq_len_features - 1:i + seq_len_features + predict_window, 0])
        if model == 'normal':
            features_list.append(torch.FloatTensor(np.array(sample_i)))
            labels_list.append(torch.LongTensor(np.array(labels_i)))
        elif model == 'InceptionTime':
            features_list += sample_i
            labels_list += labels_i
        else:
            raise ValueError("no this model {}".format(model))
        i += gen_sample_interval
    return features_list, labels_list

def generate_samples_no_cluster_feature(features, seq_len_features, predict_window, gen_sample_interval,
                                        idx, z_scored_data_dict, mem_cluster, data_ud_cluster, stock_idx_cluster,
                                        hs300_data, model='normal'):
    features_list, labels_list = [], []
    i = 0
    while i < features.shape[1]:
        if i + seq_len_features + predict_window - 1 >= features.shape[1]:
            break
        sample_i = []
        for j in range(features.shape[0]):
            # seq_len_features * 5
            hs300_feats = generate_hs300_feats(i, i + seq_len_features, hs300_data)

            tmp_data = [features[j][i:i + seq_len_features],
                        hs300_feats]
            sample_i.append(
                # [np.array([tmp_data[0][s], tmp_data[1][s]]).flatten()
                [np.array([tmp_data[0][s]]).flatten()
                 for s in range(len(tmp_data[0]))]
            )
            # 每个 sample: seq_len_features * 10
        labels_i = get_target_label(features[:, i + seq_len_features - 1:i + seq_len_features + predict_window, 0])
        if model == 'normal':
            features_list.append(torch.FloatTensor(np.array(sample_i)))
            labels_list.append(torch.LongTensor(np.array(labels_i)))
        elif model == 'InceptionTime':
            features_list += sample_i
            labels_list += labels_i
        else:
            raise ValueError("no this model {}".format(model))
        i += gen_sample_interval
    return features_list, labels_list


def generate_samples_DARNN(features, seq_len_features, predict_window, gen_sample_interval,
                     idx, z_scored_data_dict, mem_cluster, data_ud_cluster, stock_idx_cluster,
                     hs300_data):
    features_list, labels_list = [], []
    i = 0
    while i < features.shape[1]:
        if i + seq_len_features + predict_window - 1 >= features.shape[1]:
            break
        for j in range(features.shape[0]):
            # seq_len_features * 5
            # fusion_cluster_feature = generate_cluster_feats(idx[j], z_scored_data_dict,
            #                                                 i, i + seq_len_features,
            #                                                 mem_cluster, data_ud_cluster,
            #                                                 stock_idx_cluster, 'numerical_mean')
            hs300_feats = generate_hs300_feats(i, i + seq_len_features, hs300_data)

            tmp_data = [features[j][i:i + seq_len_features],
                        # fusion_cluster_feature,
                        hs300_feats]
            features_list.append(
                [np.array([tmp_data[0][s], tmp_data[1][s]]).flatten()
                 for s in range(len(tmp_data[0]))]
            )
            #labels_list.append([t[0] for t in features[j][i:i+seq_len_features]])
            this_label = [t[0] for t in features[j][i:i+seq_len_features]]
            this_label.append(features[j][i + seq_len_features + predict_window - 1][0])
            labels_list.append(this_label)
        i += gen_sample_interval
    return features_list, labels_list


def generate_samples_TPA_LSTM(features):
    features_list = []
    for i in range(features.shape[1]):
        tmp_data = []
        for j in range(features.shape[0]):
            tmp_data += [features[j][i][0],features[j][i][1],features[j][i][2],
                         features[j][i][3],features[j][i][4]]
        features_list.append(tmp_data)
    np_feature = np.array(features_list)
    np.savetxt("data/TPA_LSTM/data.txt", np_feature, delimiter=',')


def generate_samples_SFM(features):
    features_list = []
    for i in range(features.shape[0]):
        features_list.append(features[i,:,0])
    features_list = np.array(features_list)
    np.save("data/SFM/data.npy", features_list)



def load_data_and_gen_samples(dataset="zjy",
                              weighted_graph=True, weighted_graph_file='data/ssn/fixed_ssn_line.csv',
                              seq_len_features=30, predict_window=7, gen_sample_interval=1, mode='normal'):
    # 修改：每天的数据维度为3*5=15
    '''
    :param mode: what model these sampls for
    :param dataset:
    :param weighted_graph:
    :param weighted_graph_file:
    :param seq_len_features: features' length of day
    :param predict_window:
    :param gen_sample_interval:
    :return:
    '''
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
    if mode == 'normal':
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
                       'data/pickle_seq40_pwin3_intv_3/')
    elif mode == 'normal_no_cluster':
        features_list, labels_list = generate_samples_no_cluster_feature(features, seq_len_features, predict_window,
                                                                         gen_sample_interval, idx, z_scored_data_dict,
                                                                         mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                                         train_hs300)
        test_features_list, test_labels_list = generate_samples_no_cluster_feature(test_features, seq_len_features, predict_window,
                                                                gen_sample_interval, test_idx, test_z_scored_data_dict,
                                                                mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                                test_hs300)
        # features = torch.FloatTensor(np.array(features.todense()))
        # labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        save_as_pickle([adj, features_list, labels_list, test_features_list, test_labels_list],
                       ['adj', 'features_list', 'labels_list', 'test_features_list', 'test_labels_list'],
                       'data/pickle_seq30_pwin1_intv_1_no_cluster_forPaper/')

    elif mode == 'DARNN':
        features_list, labels_list = generate_samples_DARNN(features, seq_len_features, predict_window,
                                                      gen_sample_interval, idx, z_scored_data_dict,
                                                      mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                      train_hs300)
        test_features_list, test_labels_list = generate_samples_DARNN(test_features, seq_len_features, predict_window,
                                                               gen_sample_interval, test_idx, test_z_scored_data_dict,
                                                               mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                               test_hs300)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        save_as_pickle([adj, features_list, labels_list, test_features_list, test_labels_list],
                       ['adj', 'features_list', 'labels_list', 'test_features_list', 'test_labels_list'],
                       'data/pickle_seq30_pwin3_intv_3_DARNN_hasHSforPaper/')

    elif mode == 'SFM':
        generate_samples_SFM(features)

    elif mode == 'InceptionTime':
        features_list, labels_list = generate_samples_no_cluster_feature(features, seq_len_features, predict_window,
                                                      gen_sample_interval, idx, z_scored_data_dict,
                                                      mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                      train_hs300, model='InceptionTime')
        test_features_list, test_labels_list = generate_samples_no_cluster_feature(test_features, seq_len_features, predict_window,
                                                                gen_sample_interval, test_idx, test_z_scored_data_dict,
                                                                mem_cluster, data_ud_cluster, stock_idx_cluster,
                                                                test_hs300, model='InceptionTime')
        features_list = np.array(features_list)
        labels_list = np.array(labels_list)
        test_features_list = np.array(test_features_list)
        test_labels_list = np.array(test_labels_list)
        print(features_list.shape, labels_list.shape, test_features_list.shape, test_labels_list.shape)
        np.save("data/InceptionTime_forPaper/X_TRAIN.npy", features_list)
        np.save("data/InceptionTime_forPaper/Y_TRAIN.npy", labels_list)
        np.save("data/InceptionTime_forPaper/X_TEST.npy", test_features_list)
        np.save("data/InceptionTime_forPaper/Y_TEST.npy", test_labels_list)

    elif mode == "TPA_LSTM":
        generate_samples_TPA_LSTM(features)


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
    load_data_and_gen_samples(mode='normal')
    # a = torch.Tensor([0.86, 0.42, 0.33, 0.65, 0.71,0.476])
    # aaa = torch.Tensor([[0.86,0.14],[0.73,0.27],[0.69,0.31],[0.44,0.56],[0.37,0.63]])
    # ccc = torch.Tensor([[0.12,0.44]])
    # print(torch.cat((aaa,ccc),dim=0))
    # b = torch.Tensor([0,1,0,0,1])
    # c = torch.Tensor([1,0,1,1,1])
    # d = torch.Tensor([0,0,0,1,1])
    # f = ((c.data == 1) & (d.data == 1)).cpu().sum().item()
    # compute_F1(aaa,b)
    # xx = torch.FloatTensor([[[1,2,3],[5,6,7]],[[4,5,6],[5,3,1]]]).reshape((2,-1))
    # print(xx)
    # input()
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
