# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:22:16 2017

@author: damba
"""

import csv
import numpy as np

label = {}
label['M'] = 0
label['B'] = 1

def read_CSV(file):
    f = open(file)
    csvReader = csv.reader(f)
    data = []
    for row in csvReader:
        data.append(row)
    return np.array(data)

def remove_ID(data):
    return data[:,1:]

def normalize(data):
    data[:, 0] = [int(label[x]) for x in data[:, 0]]
    data = data.astype('float64')
    for j in range(1, data.shape[1]):
        x_min = min(data[:, j])
        x_max = max(data[:, j])
        data[:, j] = [1.*(x - x_min) / (x_max - x_min) for x in data[:, j]]
    return data

def split_data(data):
    validation_size = int(data.shape[0]/5)
    test_size = int(data.shape[0]/5)
    train_size = data.shape[0] - validation_size - test_size
    np.random.shuffle(data)
    train_datasets = data[0:train_size, :]
    validation_datasets = data[train_size:train_size+validation_size, :]
    test_datasets = data[train_size+validation_size:, :]
    return train_datasets, validation_datasets, test_datasets

def get_label_feature(dataset):
    return dataset[:, 0].reshape(dataset[:, 0].shape[0], 1), dataset[:, 1:]

def sigmoid(z):
    return 1./(1.+np.exp(-z))

def derivative_of_sigmoid(sigmoid_x):
	return sigmoid_x*(1-sigmoid_x)

def decision(val, T):
    if(val >= T):
        return 1
    return 0

def get_accuracy(Y_true, Y_predict):
    tmp = [1-int(Y_true[i,0])^int(Y_predict[i,0]) for i in range(len(Y_true))]
    return 1.*sum(tmp)/len(tmp)

def get_precision(Y_true, Y_predict):
    tp = sum([int(Y_true[i,0]) & int(Y_predict[i,0]) for i in range(len(Y_true))])
    fp = sum([int(1 - Y_true[i,0]) & int(Y_predict[i,0]) for i in range(len(Y_true))])
    return 1.*tp/(tp + fp)

def get_recall(Y_true, Y_predict):
    tp = sum([int(Y_true[i,0]) & int(Y_predict[i,0]) for i in range(len(Y_true))])
    fn = sum([int(Y_true[i,0]) & int(1-Y_predict[i,0]) for i in range(len(Y_true))])
    return 1.*tp/(tp + fn)

def f_score(precision, recall):
    return 2.*precision*recall/(precision+recall)

def train(X, Y):
    theta = np.zeros((X.shape[1],1))
    m = X.shape[0]
    alpha = 0.5
    print('TRAIN :: Learning rate: %f' % alpha)
    for i in range(0, 50000):
        YY = sigmoid(np.dot(X,theta))
        J = np.sum(np.square(YY-Y)*1.0/(2*m))
        theta -= alpha * np.dot(X.T,np.multiply(YY-Y, derivative_of_sigmoid(YY))) / m
        if(i % 1000 == 0):
            print('TRAIN :: Lost: %f' % J)
    print('TRAIN :: Lost: %f' % J)
    return theta

def validate(X, Y, theta):
    list_T = np.linspace(0, 1, 16)
    best_T = 0
    best_acc = 0
    for t in list_T:
        predict = sigmoid(np.dot(X,theta))
        Y_predict = np.array([decision(x, t) for x in predict]).reshape(len(predict), 1)
        acc = get_accuracy(Y, Y_predict)
        print('VALIDATE :: T(%f) <-> acc(%f)' % (t, acc))
        if(acc > best_acc):
            best_acc = acc
            best_T = t
    print('VALIDATE :: Best threshold: T = %f' % best_T)
    print('VALIDATE :: Accuracy: acc = %f' % best_acc)
    return best_T

def test(X, Y, theta, T):
    predict = sigmoid(np.dot(X,theta))
    Y_predict = np.array([decision(x, T) for x in predict]).reshape(len(predict), 1)
    acc = get_accuracy(Y, Y_predict)
    pre = get_precision(Y, Y_predict)
    re = get_recall(Y, Y_predict)
    f = f_score(pre, re)
    print('TEST :: Accuracy: %f' % acc)
    print('TEST :: Precision: %f' % pre)
    print('TEST :: Recall: %f' % re)
    print('TEST :: F-Score: %f' % f)
    

data = read_CSV('wisconsin.wdbc.data.csv')
data = remove_ID(data)
data = normalize(data)
train_datasets, validation_datasets, test_datasets = split_data(data)
Y_train, X_train = get_label_feature(train_datasets)
Y_validation, X_validation = get_label_feature(validation_datasets)
Y_test, X_test = get_label_feature(test_datasets)

theta = train(X_train, Y_train)
T = validate(X_validation, Y_validation, theta)
test(X_test, Y_test, theta, T)