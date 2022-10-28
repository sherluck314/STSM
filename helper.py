import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import seaborn as sns
import random
import warnings
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse
import time
import math
import os

def init_parser():
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset', default='Lorentz', help='name of dataset to use')
    parser.add_argument('-model', default='STSM', help='name of model to use')
    parser.add_argument('-save_name', default='', help='name of model to save')
    parser.add_argument('-device', default='0', help='GPU to use')
    parser.add_argument('-restore', action='store_true', help='restore from the previously saved model')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-cl', type=bool, default=True, help='if change learning rate while training')
    parser.add_argument('-epoch', type=int, default=300, help='training epoch')
    parser.add_argument('-bs', type=int, default=8, help='batch size for training')
    parser.add_argument('-w', type=float, default=0.0, help='weight_decay')
    parser.add_argument('-seed', type=int, default=47, help='random seed for training')
    parser.add_argument('-tpt', type=int, default=100, help='time points used for training')
    parser.add_argument('-ebl', type=int, default=10, help='embedding length used for training')
    parser.add_argument('-ttl', type=int, default=1000, help='total time length used for training')
    parser.add_argument('-tr', type=float, default=0.9, help='rate of training set')
    parser.add_argument('-dr', type=float, default=1.0, help='datarate for sampling')
    parser.add_argument('-si', type=int, default=0, help='start index for sampling')
    parser.add_argument('-ei', type=int, default=0, help='end index for sampling')
    parser.add_argument('-itv', type=int, default=1, help='interval for sampling')
    parser.add_argument('-infitv', type=int, default=5, help='interval for showing information')
    parser.add_argument('-tv', type=int, default=0, help='target variable to predict')
    parser.add_argument('-prelen', type=int, default=0, help='length of long-term prediction')
    parser.add_argument('-mm', type=int, default=0, help='merge mode for modules')
    parser.add_argument('-act', type=int, default=0, help='activate function, 0 is relu and 1 is tanh')
    parser.add_argument('-dp', type=float, default=0.0, help='dropout of model')
    parser.add_argument('--tunits', nargs='+', type=int, help="units for temporal module")
    parser.add_argument('--sunits', nargs='+', type=int, help="units for spatial module")
    parser.add_argument('--fcunits', nargs='+', type=int, help="units for fully connected layer")
    parser.add_argument('--kw', nargs='+', type=int, help="kernel widths for temporal module")
    parser.add_argument('--kn', nargs='+', type=int, help="kernel nums for temporal module")
    parser.add_argument('-bi', type=bool, default=False, help='if use bidirectional LSTM')
    return parser

def init_indicators():
    Indicators = {}
    Indicators['train_losses'] = []
    Indicators['valid_losses'] = []
    Indicators['test_losses'] = []
    return Indicators

def load_dataset(dataset):
    if dataset == "Lorentz":
        sdata = '../dataset/Lorentz/Long_Lorentz.txt'
        data = np.loadtxt(sdata)
        data = np.transpose(data)
        return data
    if dataset == "TLorentz":
        sdata = '../dataset/Lorentz/tlorentz.txt'
        data = np.loadtxt(sdata)
        return data
    if dataset == "Gene":
        sdata = '../dataset/Gene/Gene.csv'
        data = pd.read_csv(sdata)
        data = np.array(data) 
        return data
    if dataset == "wind":
        sdata = '../dataset/wind/windspeed.npy'
        data = np.load(sdata)
        data = data.reshape((-1, 6, 155))
        data = np.mean(data, axis=1)  
        return data
    if dataset == "hk":
        sdata = '../dataset/hk/hk.npy'
        data = np.load(sdata)
        return data
    if dataset == "Traffic":
        sdata = '../dataset/Traffic/traffic.npy'
        data = np.load(sdata)
        data = data.reshape((-1, 3, 33))
        data = np.mean(data, axis=1) 
        return data
    if dataset == "Plankton":
        sdata = '../dataset/Plankton/train.txt'
        data = np.loadtxt(sdata)
        sldata = '../dataset/Plankton/label.txt'
        ldata = np.loadtxt(sldata)
        return data, ldata
    if dataset == "Solar":
        sdata = '../dataset/Solar/solar_AL.txt'
        data = np.loadtxt(sdata, delimiter=',')
        return data
    if dataset == "electricity":
        sdata = '../dataset/electricity/electricity.txt'
        data = np.loadtxt(sdata, delimiter=',')
        return data

def remove_random(SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(SEED)

def show_params(params):
    for key in params.__dict__.keys():
        print(key, " "+"-"*8+" ", params.__getattribute__(key))

def get_index(stat_index, end_index, interval, train_rate, seed, datarate):
    # get index for sampling
    np.random.seed(seed)
    indexs = list(range(stat_index, end_index-interval, interval))
    np.random.shuffle(indexs)
    train_indexs = indexs[0:int(len(indexs)*train_rate*datarate)]
    test_indexs = indexs[int(len(indexs)*train_rate*datarate):int(len(indexs)*train_rate*datarate)+int(len(indexs)*(1-train_rate)*datarate)]
    return train_indexs, test_indexs


def get_sample(data, total_time_length, target_variable, train_time_length, embedding_length):
    # get sampled data
    X = data[total_time_length:total_time_length+train_time_length,:]
    Y = []
    if isinstance(target_variable, int):
        target_data = data[:, target_variable]
    else:
        target_data = target_variable
    for i in range(embedding_length):
        Y.append(target_data[total_time_length+i:total_time_length+train_time_length+i])
    Y = np.stack(Y, axis=1)
    return X,Y

def embedding_resolve(embedding):
    # resolve prediction data from embedding
    y = []
    m,s = embedding.shape
    for i in range(m+s-1):
        cur = []
        for j in range(max(0, i-s+1), min(m, i+1)):
            cur.append(embedding[j][i-j].item())
        y.append(np.mean(cur))
    return np.array(y)

def show_multi_curve(ys, title, legends, xxlabel, yylabel, if_point = False):
    # show multiple curves on a graph.
    x = np.array(range(len(ys[0])))
    for i in range(len(ys)):
        if if_point:
            plt.plot(x, ys[i], label = legends[i], marker = 'o')
        else:
            plt.plot(x, ys[i], label = legends[i])   
    plt.axis()
    plt.title(title)
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.legend()
    plt.show()

def show_curve(ys, xxlabel, yylabel, title):
    # show single curve on a graph.
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.show()

def MAE(y_true, y_pred):
    # compute mae for results
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

def RMSE(y_true, y_pred):
    # compute rmse for results
    n = len(y_true)
    arr = y_true - y_pred
    mse = 0
    for each in arr:
        mse = mse + math.pow(each, 2)
    mse = mse / n
    return math.sqrt(mse)

def compare_results(label, predict, train_time_points, model_name, data_name):
    # compare results
    label_known = label[0:train_time_points]
    predict_known = predict[0:train_time_points]
    label_unknown = label[train_time_points:]
    predict_unknown = predict[train_time_points:]
    print('test known MAE', MAE(predict_known, label_known))
    print('test known RMSE', RMSE(predict_known, label_known))
    print('test known pearsonr', pearsonr(predict_known, label_known))
    print('test unknown MAE', MAE(predict_unknown, label_unknown))
    print('test unknown RMSE', RMSE(predict_unknown, label_unknown))
    print('test unknown pearsonr', pearsonr(predict_unknown, label_unknown))
    x = np.array(range(len(label)))
    xx = np.array(range(train_time_points, len(label)))
    print(list(label))
    plt.plot(x, label, label = 'Real')
    plt.scatter(xx, predict_unknown, label = 'Linear',c='', marker = 'o', edgecolors='r')
    plt.title('Predictions of ' +model_name+ ' on '+data_name)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    print(list(predict_unknown))