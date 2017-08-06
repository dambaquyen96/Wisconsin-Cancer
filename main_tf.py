# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 09:19:00 2017

@author: damba
"""

"""
In this code, the output data is 2-dimensional
(1, 0) coressponding the M
(0, 1) coressponding the B
The classification class is the ones have higher score
The train function is flexible, works with various hidden and various size
"""

import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def read_CSV(file):
    """ Read dataset from csv file """
    f = open(file)
    csvReader = csv.reader(f)
    data = []
    for row in csvReader:
        data.append(row)
    return np.array(data)

def remove_ID(data):
    """ Remove unused feature """
    return data[:,1:]

def normalize(data):
    """ Normalize data to [0,1] """
    global label
    data[:, 0] = [int(label[x]) for x in data[:, 0]]
    data = data.astype('float64')
    for j in range(1, data.shape[1]):
        x_min = min(data[:, j])
        x_max = max(data[:, j])
        data[:, j] = [1.*(x - x_min) / (x_max - x_min) for x in data[:, j]]
    return data

def split_data(data):
    """ Split data to train, validation, test """
    validation_size = int(data.shape[0]/5)
    test_size = int(data.shape[0]/5)
    train_size = data.shape[0] - validation_size - test_size
    np.random.shuffle(data)
    train_datasets = data[0:train_size, :]
    validation_datasets = data[train_size:train_size+validation_size, :]
    test_datasets = data[train_size+validation_size:, :]
    return train_datasets, validation_datasets, test_datasets

def get_label_feature(dataset):
    """ Split label & feature """
    single_label = dataset[:, 0]
    label = np.array([[x, 1-x] for x in single_label])
    return label, dataset[:, 1:]

class Model:
    def __init__(self, weights):
        self.weights = weights

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forward(x, w):
    """ Forward propagation """
    h_size = len(w)-1
    h = [None] * h_size
    
    h[0] = tf.nn.sigmoid(tf.matmul(x, w[0]))
    for i in range(1, h_size):
        h[i] = tf.nn.sigmoid(tf.matmul(h[i-1], w[i]))
    y_hat = tf.matmul(h[h_size-1], w[h_size])
    return y_hat
    
def plot_loss(name, J):
    plt.plot(J)
    plt.title(name)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

def train(X, Y, X_val, Y_val, h_size, learning_rate=0.05):
    """ Tranining with various hidden layer """
    # Layer size
    num_hl = len(h_size)
    x_size = X.shape[1]
    y_size = Y.shape[1]
    w_size = len(h_size) + 1
    
    print("[TRAIN-%dHL] Trainint model with %d hidden layer" % (num_hl, num_hl))
    
    with tf.Session() as session:
        # Inputs
        x = tf.placeholder(name='x', dtype="float", shape=(None, x_size))
        y = tf.placeholder(name='y', dtype="float", shape=(None, y_size))
        
        # Params
        w = [None]*w_size
        for i in range(w_size):
            if(i == 0):
                w[i] = init_weights((x_size, h_size[i]))
            elif(i == w_size-1):
                w[i] = init_weights((h_size[i-1], y_size))
            else:
                w[i] = init_weights((h_size[i-1], h_size[i]))
        
        # Forward
        y_hat = forward(x, w)
        predict = tf.argmax(y_hat, axis=1)
        
        # Backward
        J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(J)
        
        # Init params
        init = tf.global_variables_initializer()
        session.run(init)
        
        # Running tensorflow
        step = 0
        old_loss = None
        J_step = []
        while True:
            loss, _ = session.run([J, opt], feed_dict={x:X, y:Y})
            J_step.append(loss)
            step += 1
            if(step % 1000 == 0):
                loss_val = session.run(J, feed_dict={x:X_val, y:Y_val})
                print("[TRAIN-%dHL] Step %5d: Train_loss = %f --- Validation_loss = %f" % (num_hl, step, loss, loss_val))
                if(old_loss != None and loss_val >= old_loss):
                    break
                old_loss = loss_val
        
        # Get accuracy
        acc = np.mean(np.argmax(Y_val, axis=1) == session.run(predict, feed_dict={x:X_val, y:Y_val}))
        return Model([wi.eval() for wi in w]), acc, J_step
        
def validate(models, hidden_layer, accs, J):
    """ Using validation dataset to find the best model """
    layers = None
    model_final = None
    best_acc = -1
    for i in range(len(models)):
        print("[VALIDATION] %d hidden layer: acc = %f" % (len(hidden_layer[i]), accs[i]))
        plot_loss(str(len(hidden_layer[i])) + ' hidden layer', J[i])
        if(accs[i] > best_acc):
            best_acc = accs[i]
            model_final = models[i]
            layers = hidden_layer[i]
    print("[VALIDATION] Final model: %d hidden layer with acc = %f" % (len(layers), best_acc))
    return model_final, layers, best_acc

def test(X, Y, W):
    """ Testing model with test dataset """
    with tf.Session() as session:
        x = tf.placeholder(name='x', dtype="float", shape=(None, X.shape[1]))
        w = [tf.placeholder(dtype="float", shape=(Wi.shape[0], Wi.shape[1])) for Wi in W]
        y_hat = forward(x, w)
        predict = tf.argmax(y_hat, axis=1)
        
        init = tf.global_variables_initializer()
        session.run(init)

        fd = {wi:Wi for wi,Wi in zip(w,W)}
        fd.update({x:X})
        y_predict = session.run(predict, feed_dict=fd)
        y = np.argmax(Y, axis=1)
        
        acc = np.mean(y == y_predict)
        tp = np.sum((y==y_predict) & (y_predict==1))
        fp = np.sum((y!=y_predict) & (y_predict==1))
        fn = np.sum((y!=y_predict) & (y_predict==0))
        pre = 1.*tp/(tp + fp)
        re = 1.*tp/(tp + fn)
        f_score = 2.*pre*re/(pre+re)
        
        print("[TEST] Accuracy : %f" % acc)
        print("[TEST] Precision : %f" % pre)
        print("[TEST] Recall : %f" % re)
        print("[TEST] F-Score : %f" % f_score)
        
        

if __name__ == "__main__":
    label = {}
    label['M'] = 1
    label['B'] = 0
    
    data = read_CSV('wisconsin.wdbc.data.csv')
    data = remove_ID(data)
    data = normalize(data)
    
    train_datasets, validation_datasets, test_datasets = split_data(data)
    Y_train, X_train = get_label_feature(train_datasets)
    Y_validation, X_validation = get_label_feature(validation_datasets)
    Y_test, X_test = get_label_feature(test_datasets)

    hidden_layer = [[4], [4,4], [4,4,3], [4,4,3,3], [4,4,4,3,3]]
    models = [None] * len(hidden_layer)
    accs = [None] * len(hidden_layer)
    J = [None] * len(hidden_layer)
    
    for i in range(len(hidden_layer)):
        models[i], accs[i], J[i] = train(X_train, Y_train, X_validation, Y_validation, hidden_layer[i])
    model_final, layers, best_acc = validate(models, hidden_layer, accs, J)
    test(X_test, Y_test, model_final.weights)
    
    