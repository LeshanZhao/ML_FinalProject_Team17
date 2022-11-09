# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:50:10 2022

@author: heckenna
"""

import numpy as np
import pandas as pd
import data_opener
from perceptron import *

data = data_opener.get_data()

# Max corr is .5, so correlation not going to be used for feature selection
corr_matrix = data.corr()
print(corr_matrix)

X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)



size = 13
h_num = 2000

# Running on top 2000 rows currently, just to save time with debugging
perc = Perceptron(size = size, epochs = 10)
perc.train_classifier(X_train.head(h_num), y_train.head(h_num))

pred_train = perc.predict_data(X_train)
pred_test = perc.predict_data(X_test)

acc_train = sum(pred_train == y_train)/len(y_train)
acc_test = sum(pred_test == y_test)/len(y_test)

print(acc_train)
print(acc_test)
