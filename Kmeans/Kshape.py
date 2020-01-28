import numpy as np
import os
from sklearn import preprocessing
import csv
import math
import random
from matplotlib import pyplot as plt
from numpy.fft import fft,ifft
from numpy import conj
#加上 0 1 -1 三分类的序列，centroid的计算方法仍然保持不变
time_interval = 'only_cluster_200106'
datadir = 'cluster_ans/for_train/'+time_interval+'/z_scored_'+time_interval+'.csv'
#data_ud_dir = 'cluster_ans/for_train/'+time_interval+'/up_and_down_'+time_interval+'.csv'
stock_idx_file = 'cluster_ans/for_train/'+time_interval+'/stock_idx_'+time_interval+'.txt'
ans_dir = 'cluster_ans/train_ans/'
#三分类序列的影响因子：不限制在0-1
Alpha = 1
def NCCc(w,m,x,y):  # y is aligned towards x
    k = w - m
    if k >= 0:
        t_sum = 0
        for i in range(m - k):
            t_sum += x[i + k] * y[i]
        if np.linalg.norm(x,ord=2)<=1e-10 or np.linalg.norm(y,ord=2)<=1e-10:
            t_sum = 0
        else:
            t_sum = t_sum/math.sqrt(np.linalg.norm(x,ord=2)*np.linalg.norm(y,ord=2))
        return t_sum
    else:
        t_sum = 0
        for i in range(m + k):
            t_sum += y[i-k] * x[i]
        if np.linalg.norm(x, ord=2) <= 1e-10 or np.linalg.norm(y, ord=2) <= 1e-10:
            t_sum = 0
        else:
            t_sum = t_sum / math.sqrt(np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2))
        return t_sum

def NCCc_fft(x,y):
    lens = len(x)
    fftlen = 2**math.ceil(math.log2(2*lens-1))
    r = ifft(fft(x,int(fftlen))*conj(fft(y,int(fftlen))))
    r_end = len(list(r))-1
    r = list(r)[r_end-lens+2:] + list(r)[:lens]
    cc_sequence = r / (np.linalg.norm(x) * np.linalg.norm(y))
    cc_sequence = [t.real for t in cc_sequence]
    return cc_sequence

def SBD_old(x,y):
    m = len(x)
    NCCc_seq = []
    for i in range(1,2*m):
    #for i in range(max(1, m - 3), min(2 * m, m + 3)):
        NCCc_seq.append(NCCc(i,m,x,y))
    value = max(NCCc_seq)
    #index = NCCc_seq.index(max(NCCc_seq))+max(1,m-3)
    index = NCCc_seq.index(max(NCCc_seq)) + 1
    dist = 1 - value
    shift = index - m
    y_ = []
    if shift >= 0:
        for t in range(shift):
            y_.append(0)
        for t in range(m-shift):
            y_.append(y[t])
    else:
        for t in range(-shift,m):
            y_.append(y[t])
        for t in range(-shift):
            y_.append(0)
    return dist,shift,y_

def SBD(x,y):  #依然默认序列等长假设
    m = len(x)
    X1 = NCCc_fft(x,y)
    value = max(X1)
    index = X1.index(value)+1
    dist = 1 - value
    shift = index - m
    y_ = []
    if shift >= 0:
        for t in range(shift):
            y_.append(0)
        for t in range(m-shift):
            y_.append(y[t])
    else:
        for t in range(-shift,m):
            y_.append(y[t])
        for t in range(-shift):
            y_.append(0)
    return dist,shift,y_

def Simmilar_up_and_down(x,cent):
    ud_cent = []
    for i in range(len(cent)-1):
        if cent[i+1]>cent[i]:
            ud_cent.append(1)
        elif cent[i+1]<cent[i]:
            ud_cent.append(-1)
        else:
            ud_cent.append(0)
    ret = np.dot(x,ud_cent) / len(x)        #[-1,1]取值
    return ret

def kshape_centroid(mem,A,k,cent_vector):  #A: z- normalized
    data_k=[]
    for i in range(len(mem)):
        if mem[i] == k:
            if math.fabs(np.sum(cent_vector))==0:
                opt_v = [A[i][t] for t in range(A.shape[1])]
                #print(len(opt_v),"1")
            else:
                tmp,tmp1,opt_v = SBD(cent_vector,A[i])
                #print(len(opt_v),"2")
            data_k.append(opt_v)
    if len(data_k) == 0:
        return np.zeros(A.shape[1])
    data_k = np.array(data_k)
    data_k = np.array(preprocessing.scale(data_k.T.tolist())).T
    S = np.dot(data_k.T,data_k)
    Q = np.eye(data_k.shape[1]) - (1/data_k.shape[1]) * np.ones((data_k.shape[1],data_k.shape[1]))
    M = np.dot(Q,S)
    M = np.dot(M,Q)
    #如果用一阶差分序列的话，就不用标准化向量 QT S Q 也就变为S
    #M = S
    e_val_s,e_vec_s = np.linalg.eig(M)
    e_val_s = list(e_val_s)
    #print(e_val_s)
    #e_vec_s = list(e_vec_s)
    e_valm = max(e_val_s)
    e_vec = e_vec_s[:,e_val_s.index(max(e_val_s))]
    e_vec = [i.real for i in e_vec]
    dis_1 = [data_k[0][i]-e_vec[i] for i in range(data_k.shape[1])]
    dis_2 = [data_k[0][i]+e_vec[i] for i in range(data_k.shape[1])]
    if np.linalg.norm(dis_1) >= np.linalg.norm(dis_2):
        e_vec = [-t for t in e_vec]
    e_vec = preprocessing.scale(e_vec)
    #一阶差分序列不做标准化
    return e_vec

def Kshape(X,K): #X:m*n(numpy) matrix    k:number of clusters
    m = X.shape[0]
    n = X.shape[1]
    iter = 0
    mem = list(np.random.rand(m))
    mem = [i*K for i in mem]
    mem = [int(math.floor(i)) for i in mem]   #index:which cluster the data should belong to
    cent = np.zeros((K,n),dtype=float)
    while iter <= 1000:
        print(iter)
        print(mem)
        prev_mem = [t for t in mem]          #deep copy
        for i in range(K):
            cent[i] = kshape_centroid(mem,X,i,cent[i])
        D = np.zeros((m,K),dtype=float)
        for i in range(m):
            for j in range(K):
                d_,t1,t2 = SBD(X[i],cent[j])
                D[i][j] = d_
        D = D.tolist()
        mem = [t.index(min(t)) for t in D]
        if np.linalg.norm([(prev_mem[i]-mem[i]) for i in range(len(mem))]) == 0:
            break
        iter += 1
    return mem,cent


def main(num_cluster, ans_file_name, idx_file, data_dir):
    Data = []
    with open(data_dir, 'r') as f:
        Data = list(csv.reader(f))
    Data = np.array(Data,dtype=float)
    mem, cent = Kshape(Data, num_cluster)
    # plot and write ans to files
    mem_file = 'mem.csv'
    cent_file = 'cent.csv'
    for tt in range(num_cluster):
        data_1 = []
        plt.figure()
        idx = [i for i in range(1, Data.shape[1] + 1)]
        for i in range(Data.shape[0]):
            if mem[i] == tt:
                plt.plot(idx, Data[i])
        plt.show()

    stock_idx = []
    with open(idx_file, 'r') as f:
        for line in f.readlines():
            stock_idx.append(line.replace('\n', ''))
    for i, tmp in enumerate(stock_idx):
        stock_idx[i] = [stock_idx[i], mem[i]]
    if not os.path.exists(ans_dir):
        os.mkdir(ans_dir)

    concret_ans_dir = os.path.join(ans_dir, ans_file_name + '_' + str(num_cluster))
    with open(concret_ans_dir + '_' + mem_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(stock_idx)

    with open(concret_ans_dir+ '_' + cent_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for ct in range(cent.shape[0]):
            writer.writerow([ct] + list(cent[ct]))


if __name__ == '__main__':
    main(20, "only_cluster_200106", stock_idx_file, datadir)
    # Data = []
    # num_cluster = 10
    # with open(datadir,'r') as f:
    #     Data = list(csv.reader(f))
    # Data = np.array(Data,dtype=float)
    # '''
    # Data_ud = []
    # with open(data_ud_dir, 'r') as f:
    #     Data_ud = list(csv.reader(f))
    # Data_ud = np.array(Data_ud, dtype=float)
    # '''
    # mem,cent = Kshape(Data,num_cluster)
    #
    # #plot and write ans to files
    # mem_file = 'mem_'+time_interval+'.csv'
    # cent_file = 'cent_'+time_interval+'.csv'
    #
    # for tt in range(num_cluster):
    #     data_1 = []
    #     plt.figure()
    #     idx = [i for i in range(1,Data.shape[1]+1)]
    #     for i in range(Data.shape[0]):
    #         if mem[i] == tt:
    #             plt.plot(idx,Data[i])
    #     plt.show()
    #
    # stock_idx = []
    # with open(stock_idx_file,'r') as f:
    #     for line in f.readlines():
    #         stock_idx.append(line.replace('\n',''))
    # for i,tmp in enumerate(stock_idx):
    #     stock_idx[i] = [stock_idx[i],mem[i]]
    # if not os.path.exists(ans_dir):
    #     os.mkdir(ans_dir)
    # with open(ans_dir+mem_file,'w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(stock_idx)
    #
    # with open(ans_dir+cent_file,'w',newline='') as f:
    #     writer = csv.writer(f)
    #     for ct in range(cent.shape[0]):
    #         writer.writerow([ct]+list(cent[ct]))




