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
from mlp import MLP

random.seed(12321)

data = data_opener.get_data()

# Max corr is .5, so correlation not going to be used for feature selection
corr_matrix = data.corr()
#print(corr_matrix)

X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)


size = 13
h_num = 500





# Running on top 500 rows currently, just to save time with debugging

# TODO: Test NN functionality

iden = np.identity(n = size)
layer_in_test = layer.Layer(num_perceptrons = size,
                            num_inputs = size, 
                            is_input_layer = True)
layer_h_test = layer.Layer(num_perceptrons = 2, 
                            num_inputs = size)
layer_o_test = layer.Layer(num_perceptrons = 1, 
                            num_inputs = 2)

mlp_1 = MLP(num_features=size,
            num_hidden_layers=1,
            hidden_sizes=[2])


"""
# TODO: add number of epochs and lr as NN param...

#print(X_train.iloc[1])
x_val = X_train.iloc[1]
y_val = y_train.iloc[1]
lr = .5


for e in range(20):
    for i in range(100):
        x_val = X_train.iloc[i]
        y_val = y_train.iloc[i]
        
        lay_out = layer_in_test.forward(x_val)
        lay_out2 = layer_h_test.forward(lay_out)
        lay_out3 = layer_o_test.forward(lay_out2)
    
    
        d_o = layer_o_test.backward(lr = lr, y_train = y_val)
        d_h = layer_h_test.backward(lr, next_deltas = d_o, next_weights = layer_o_test.weight_matrix)
        #d_i = layer_in_test.backward(lr, 
        #                     next_deltas = d_h, 
        #                     next_weights = layer_h_test.weight_matrix)



pred_y = []

y_head = [y_train.iloc[i] for i in range(100)]

for i in range(100):
    x_val = X_train.iloc[i]
    y_val = y_train.iloc[i]
    
    lay_out = layer_in_test.forward(x_val)
    lay_out2 = layer_h_test.forward(lay_out)
    lay_out3 = layer_o_test.forward(lay_out2)

    if lay_out3[0] >= .5:        
        pred_y.append(1)
    else:
        pred_y.append(0)
        
        
y_head = np.array(y_head)
pred_y = np.array(pred_y)

def get_acc(pred, y_test):
    return np.sum(y_test == pred) / len(y_test) * 100

print(get_acc(pred_y, y_head))

#"""

'''x_val = X_train.iloc[2]
y_val = y_train.iloc[2]
lr = 5
# TODO: For whatever reason, we are gravitating towards y=.5 ...?

lay_out = layer_in_test.forward(x_val)
lay_out2 = layer_h_test.forward(lay_out)
lay_out3 = layer_o_test.forward(lay_out2)

d_o = layer_o_test.backward(lr = lr, y_train = y_val)
d_h = layer_h_test.backward(lr, 
                            next_deltas = d_o, 
                            next_weights = layer_o_test.weight_matrix)
d_i = layer_in_test.backward(lr, 
                             next_deltas = d_h, 
                             next_weights = layer_h_test.weight_matrix)

print(lay_out)
print(lay_out2)
print(lay_out3)'''

#perc = Perceptron(size = size, epochs = 10)
#perc.train_classifier(X_train.head(h_num), y_train.head(h_num))

#pred_train = perc.predict_data(X_train)
#pred_test = perc.predict_data(X_test)

#acc_train = sum(pred_train == y_train)/len(y_train)
#acc_test = sum(pred_test == y_test)/len(y_test)

#print(acc_train)
#print(acc_test)



