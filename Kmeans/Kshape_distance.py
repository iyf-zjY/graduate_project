import tensorflow as tf
import numpy as np
from numpy.fft import fft,ifft
from numpy import conj
from sklearn import preprocessing
import math

a = [1,2,3,4,5,6,7]
b = [4,3,2,1,7,6,8]

def NCCc_fft(x,y):
    lens = len(x)
    fftlen = 2**math.ceil(math.log2(2*lens-1))
    r = ifft(fft(x,int(fftlen))*conj(fft(y,int(fftlen))))
    r_end = len(list(r))-1
    r = list(r)[r_end-lens+2:] + list(r)[:lens]
    cc_sequence = r / (np.linalg.norm(x) * np.linalg.norm(y))
    cc_sequence = [t.real for t in cc_sequence]
    return cc_sequence

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


def NCCc(w,m,x,y):  # y is aligned towards x
    k = w - m
    if k >= 0:
        t_sum = 0
        for i in range(m - k):
            t_sum += x[i + k] * y[i]
        if np.linalg.norm(x,ord=2)<=1e-10 or np.linalg.norm(y,ord=2)<=1e-10:   #之前的算法好像写错了，二范数不用再开方了
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

def SBD_t(x,y):
    m = len(x)
    NCCc_seq = []
    for i in range(1, 2 * m):
        # for i in range(max(1, m - 3), min(2 * m, m + 3)):
        NCCc_seq.append(NCCc(i, m, x, y))
    print(NCCc_seq)


if __name__ == '__main__':
    a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
    b = a[1]
    with tf.Session() as sess:
        print(sess.run(b))

