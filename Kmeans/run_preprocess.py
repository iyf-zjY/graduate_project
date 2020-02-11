import pymysql
import numpy as np
import math
import os
import csv
import pandas as pd
import datetime
from sklearn import preprocessing
global stock_idx
from xml.etree import ElementTree as ET

attr='close'
#time_interval = ET.parse(conf).getroot()[1].text.strip()
time_interval = '190601-190930'
data_dir = 'data/price_data/'+time_interval+'/hs300/'

def read_data(attr):
    with open(data_dir+attr+'.csv','r') as f:
        reader = list(csv.reader(f))[1:]
        return reader
#z-scored
def preprocess(data):
    data_scaled = preprocessing.scale(data)
    return data_scaled

#1st_difference
def preprocess_1st_dif(data):
    ret_data = []
    for s in data:
        d_s = [(float(s[i+1])-float(s[i])) for i in range(len(s)-1)]
        d_s = [(1 if t > 0 else -1) for t in d_s]
        ret_data.append(d_s)
    return ret_data


def main():
    Data = read_data(attr)
    stock_idx = [it[0] for it in Data]
    Data = [it[1:] for it in Data]
    Data_triple = preprocess_1st_dif(Data)  # 0 1 -1 三分类序列

    Data = (np.array(Data).T).tolist()
    Data = (np.array(preprocess(Data)).T).tolist()

    # Data = preprocess_1st_dif(Data)
    out_dir = data_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(out_dir + 'z_scored_' + time_interval + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(Data)
    with open(out_dir + 'stock_idx_' + time_interval + '.txt', 'w') as f:
        for s in stock_idx:
            f.write(s + '\n')
    with open(out_dir + 'up_and_down_' + time_interval + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(Data_triple)


if __name__ == '__main__':
    pass
