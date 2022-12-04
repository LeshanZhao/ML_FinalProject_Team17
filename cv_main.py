# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:57:02 2022

@author: heckenna
"""

import numpy as np
import pandas as pd
import data_opener
from cross_validation import cross_validation
import mlp
import random
#from sklearn.model_selection import train_test_split

def make_ones_and_zeroes(col):
    return np.vectorize(lambda val: 1 if (val >= .5) else 0)(col)

random.seed(9723)

data = data_opener.get_data()

#X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)

cv = cross_validation(data.head(100))


folds = cv.Kfold(n_splits = 4)


targ_name = "cardio"
lr = .1
num_epochs = 10

size = 13

splits = 4

for i in range(splits):
    for j in range(splits):
        if (i == 0 and j!= 0) or (j == 0 and i == 1):
            x_train = folds[i].drop(columns = targ_name)
            y_train = folds[i][targ_name]
        
        elif i == j:
            x_test = folds[i].drop(columns = targ_name)
            y_test = folds[i][targ_name]
            
        else:
            x_train = pd.concat([x_train, folds[i].drop(columns = targ_name)])
            y_train = pd.concat([y_train, folds[i][targ_name]])
        
    # Do stuff with x_train and y_train...
    neural_net = mlp.MLP(n_features = size, hidden_sizes = [8], include_bias = True)
    
    print("Started train")
    neural_net.train(x_train, y_train, epochs  = num_epochs, lr = lr)
    print("Started pred")
    y_train_pred = neural_net.pred(x_train) 
    y_test_pred = neural_net.pred(x_test) 
    
    y_train_pred = make_ones_and_zeroes(y_train_pred)
    y_test_pred = make_ones_and_zeroes(y_test_pred)
    
    print(cv.confusion_matrix(y_train_pred, y_train))
    print(cv.confusion_matrix(y_test_pred, y_test))
    
    cv.add_metrics(fold_nums = splits, y_pred = y_test_pred, y_test = y_test)
    
print("Precision:", cv.precision)
print("Recall", cv.recall)
print("Accuracy", cv.accuracy)
print("Error", cv.error)


    
    
    
    




#neural_net.train(Xc, yc, epochs  = 1, lr = lr)


