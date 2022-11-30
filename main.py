# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:50:10 2022

@author: heckenna
"""

import numpy as np
import pandas as pd
import data_opener
import layer
from perceptron import *
import random
import mlp
import cProfile
import pstats
import io
from pstats import SortKey
import matplotlib.pyplot as plt

def get_acc(pred, y):
    return (1 - np.sum(abs(np.array(y) - np.array(pred))) / len(y)) * 100

'''
ob = cProfile.Profile()
ob.enable()
'''
random.seed(12321)

data = data_opener.get_data()
# Max corr is .5, so correlation not going to be used for feature selection
corr_matrix = data.corr()
#print(corr_matrix)

X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)


n_samples = 1000
care_for = 20


size = 13

bs = 25

X0 = X_train[np.logical_and(y_train == 0, np.logical_and(X_train["active"] == 0, X_train["smoke"] == 1))].head(n_samples) #[y_train == 0]
y0 = y_train[np.logical_and(y_train == 0, np.logical_and(X_train["active"] == 0, X_train["smoke"] == 1))].head(n_samples)


X1 = X_train[np.logical_and(y_train == 1, np.logical_and(X_train["active"] == 1, X_train["smoke"] == 0))].head(n_samples) #[y_train == 0]
y1 = y_train[np.logical_and(y_train == 1, np.logical_and(X_train["active"] == 1, X_train["smoke"] == 0))].head(n_samples)

Xc = pd.concat([X0, X1])
yc = pd.concat([y0, y1])

X_try = X_train.head(n_samples)
y_try = y_train.head(n_samples)

lr = .01
n_epochs = 50


my_new_perceptron = mlp.MLP(num_features = size, 
                            num_hidden_layers = 1, 
                            hidden_sizes = [8], 
                            n_epochs = n_epochs,
                            include_bias = True)

loss_vec = []   # for plotting

# for i in range(n_epochs):
#     y_pred = my_new_perceptron.train(X_try, y_try, epochs  = 1, lr = lr)
#     losses = my_new_perceptron.loss(X_try, y_try)
#     #print(losses)
#     loss_vec.append(losses)

my_new_perceptron.train(X_try, y_try, lr = lr) 
y_pred = my_new_perceptron.pred(X_try)
# losses = my_new_perceptron.loss(X_try, y_try)

print(y_pred)
out2 = list(map(lambda x: 1 if x >= .5 else 0, y_pred))

for o, y_i in zip(y_pred, y_try):
    print(str(o), "", str(y_i))


print("Accuracy:", get_acc(out2, y_try))


# plt.plot(range(n_epochs), loss_vec)
# plt.show()
