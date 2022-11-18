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

def get_acc(pred, y):
    return (1 - np.sum(abs(np.array(y) - np.array(pred))) / len(y)) * 100


ob = cProfile.Profile()
ob.enable()


random.seed(12321)

data = data_opener.get_data()

# Max corr is .5, so correlation not going to be used for feature selection
corr_matrix = data.corr()
#print(corr_matrix)

X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)


size = 13
h_num = 2000
care_for = 20

# Running on top 2000 rows currently, just to save time with debugging
"""
# TODO: Test NN functionality
#iden = np.identity(n = size)
layer_in_test = layer.Layer(size, size, is_input_layer = True)
layer_h_test = layer.Layer(num_perceptrons = 6, num_inputs = size)
layer_h2_test = layer.Layer(num_perceptrons = 3, num_inputs = 6)
layer_o_test = layer.Layer(num_perceptrons = 1, num_inputs = 3)

# TODO: add number of epochs and lr as NN param...

#print(X_train.iloc[1])
x_val = X_train.iloc[1]
y_val = y_train.iloc[1]
lr = 5


for e in range(20):
    for i in range(100):
        x_val = X_train.iloc[i]
        y_val = y_train.iloc[i]
        
        lay_out = layer_in_test.forward(x_val)
        lay_out2 = layer_h_test.forward(lay_out)
        lay_out3 = layer_h2_test.forward(lay_out2)
        lay_out4 = layer_o_test.forward(lay_out3)
    
    
        d_o = layer_o_test.backward(lr = lr, y_train = y_val)
        d_h = layer_h2_test.backward(lr, next_deltas = d_o, next_weights = layer_o_test.weight_matrix)
        d_h2 = layer_h_test.backward(lr, next_deltas = d_h, next_weights = layer_h2_test.weight_matrix)
        
        d_i = layer_in_test.backward(lr, 
                             next_deltas = d_h2, 
                             next_weights = layer_h_test.weight_matrix)



pred_y = []

y_head = [y_train.iloc[i] for i in range(100)]

for i in range(100):
    x_val = X_train.iloc[i]
    y_val = y_train.iloc[i]
    
    lay_out = layer_in_test.forward(x_val)
    lay_out2 = layer_h_test.forward(lay_out)
    lay_out3 = layer_h2_test.forward(lay_out2)
    lay_out4 = layer_o_test.forward(lay_out3)

    if lay_out3[0] >= .5:        
        pred_y.append(1)
    else:
        pred_y.append(0)
        
        
y_head = np.array(y_head)
pred_y = np.array(pred_y)



print(get_acc(pred_y, y_head))

#"""
'''
x_val = X_train.iloc[1]
y_val = y_train.iloc[1]
lr = 5
# TODO: For whatever reason, we are gravitating towards y=.5 ...?

lay_out = layer_in_test.forward(x_val)
lay_out2 = layer_h_test.forward(lay_out)
lay_out3 = layer_h2_test.forward(lay_out2)
lay_out4 = layer_o_test.forward(lay_out3)

d_o = layer_o_test.backward(lr = lr, y_train = y_val)
d_h = layer_h2_test.backward(lr, next_deltas = d_o, next_weights = layer_o_test.weight_matrix)
d_h2 = layer_h_test.backward(lr, next_deltas = d_h, next_weights = layer_h2_test.weight_matrix)
     
d_i = layer_in_test.backward(lr, 
                             next_deltas = d_h, 
                             next_weights = layer_h_test.weight_matrix)

print(lay_out)
print(lay_out2)
print(lay_out3)
print(lay_out4)
#'''
"""
n = 7

h1_size = 5

x_val = X_train.iloc[n]
y_val = y_train.iloc[n]
lr = 5

#layer_i_test = layer.Layer(size, size, is_input_layer = True)
#layer_h1_test = layer.Layer(num_perceptrons = h1_size, num_inputs = size)
#layer_o_test = layer.Layer(num_perceptrons = 1, num_inputs = h1_size)

for i in range(200):
    for j in range(care_for):
        x_val = X_train.iloc[j]
        y_val = y_train.iloc[j]
        lay_out = layer_i_test.forward(x_val)
        #print(lay_out)
        lay_out = layer_h1_test.forward(lay_out)
        #print(lay_out)
        lay_out = layer_o_test.forward(lay_out)
        #print("Prediction:", lay_out)
        #print("Actual:", y_val)
        
        
        d_o = layer_o_test.backward(lr = lr, y_train = y_val)
        d_h1 = layer_h1_test.backward(lr, 
                                    next_deltas = d_o, 
                                    next_weights = layer_o_test.weight_matrix)
        d_i = layer_i_test.backward(lr, 
                                    next_deltas = d_h1, 
                                    next_weights = layer_h1_test.weight_matrix)
#"""
"""
pred_y = []

y_head = [y_train.iloc[i] for i in range(care_for)]

for i in range(care_for):
    x_val = X_train.iloc[i]
    y_val = y_train.iloc[i]
    
    lay_out = layer_i_test.forward(x_val)
    lay_out = layer_h1_test.forward(lay_out)
    lay_out = layer_o_test.forward(lay_out)

    if lay_out[0] >= .5:        
        pred_y.append(1)
    else:
        pred_y.append(0)

print(get_acc(pred_y, y_head))
#"""
#print(d_o)

#perc = Perceptron(size = size, epochs = 10)
#perc.train_classifier(X_train.head(h_num), y_train.head(h_num))

#pred_train = perc.predict_data(X_train)
#pred_test = perc.predict_data(X_test)

#acc_train = sum(pred_train == y_train)/len(y_train)
#acc_test = sum(pred_test == y_test)/len(y_test)
"""
#print(acc_train)
#print(acc_test)
h1_size = 5
lr = 1
#layer_i_test = layer.Layer(size, size, is_input_layer = True)
#layer_h1_test = layer.Layer(num_perceptrons = h1_size, num_inputs = size)
#layer_o_test = layer.Layer(num_perceptrons = 1, num_inputs = h1_size)

for j in range(care_for):
    x_val = X_train.iloc[j]
    y_val = y_train.iloc[j]
    lay_out = layer_i_test.forward(x_val)
    #print(lay_out)
    lay_out = layer_h1_test.forward(lay_out)
    #print(lay_out)
    lay_out = layer_o_test.forward(lay_out)
    print("Prediction:", lay_out)
    print("Actual:", y_val)
    
    
    d_o = layer_o_test.backward(lr = lr, y_train = y_val)
    d_h1 = layer_h1_test.backward(lr, 
                                next_deltas = d_o, 
                                next_weights = layer_o_test.weight_matrix)
    d_i = layer_i_test.backward(lr, 
                                next_deltas = d_h1, 
                                next_weights = layer_h1_test.weight_matrix)
#"""
size = 13
x_val = X_train.iloc[j]
y_val = y_train.iloc[j]


#smoke_col = X_train[["smoke", "active"]]

bs = 25

X_try = X_train.head(50) #[y_train == 0]
y_try = y_train.head(50)

#x_smoke = smoke_col.head(500)
#y_try = y_train.head(500)

lr = .5

#my_little_perceptron.print_network()
my_new_perceptron = mlp.MLP(num_features = size, num_hidden_layers = 1, hidden_sizes = [6])

#my_new_perceptron = mlp.MLP(num_features = size, num_hidden_layers = 1, hidden_sizes = [3])


out = my_new_perceptron.train(X_train.head(100), y_train.head(100), epochs  = 1, lr = lr, batch_size = len(y_try))

#out = my_new_perceptron.pred(X_train) #, y_try)

#out_test = my_new_perceptron.pred(X_test.head(100)) #, y_try)


#print("Output:",out)
#print("Actual:\n" + str(y_train))
#print("Test Output:", out_test)
#print("Test Actual:\n" + str(y_test))

#print(get_acc(out, y_train))


ob.disable()
sec = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(ob, stream=sec).sort_stats("tottime")
ps.print_stats(15)
ps = pstats.Stats(ob, stream=sec).sort_stats("cumtime")
ps.print_stats(15)


print(sec.getvalue())