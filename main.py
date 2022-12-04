# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:50:10 2022

@author: heckenna
"""

import numpy as np
import pandas as pd
import data_opener
import random
import mlp
import matplotlib.pyplot as plt
from cross_validation import cross_validation

def get_acc(pred, y):
    return (1 - np.sum(abs(np.array(y) - np.array(pred))) / len(y)) * 100


random.seed(12321)

data = data_opener.get_data()
# Max corr is .5, so correlation not going to be used for feature selection
corr_matrix = data.corr()

X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)


n_samples = 3000

X_train = X_train.head(n_samples)
y_train = y_train.head(n_samples)
print(X_train.iloc[0])

n_features = 13
lr = .000001
n_epochs = 600
batch_size = 25

output_path = "output/lr" + str(lr) + "_" + \
                str(n_epochs) + "epochs" + "_" + \
                "batchSize" + str(batch_size) + "_" + \
                str(n_samples) + "samples.txt"
file_result = open(output_path, "a+")

print("sample size: ", n_samples, file = file_result)
print("lr: ", lr, file = file_result)
print("n_epochs: ", n_epochs, file = file_result)
print("batch_size: ", batch_size, file = file_result)

print("=============== Start training: ===============", file = file_result)

mlp_clf = mlp.MLP(n_features = n_features, 
                    hidden_sizes = [8], 
                    n_epochs = n_epochs,
                    batch_size = batch_size,
                    include_bias = True)

loss_vec = []   # for plotting

mlp_clf.train(X_train, y_train, lr = lr)


# Use Training set
y_pred = mlp_clf.pred(X_train)
losses = mlp_clf.losses


out2 = list(map(lambda x: 1 if x >= .5 else 0, y_pred))
print("Accuracy:", get_acc(out2, y_train), file = file_result)
print("losses: ", losses, file = file_result)




for y_pred_i, y_train_i in zip(y_pred, y_train):
    print(str(y_pred_i), "", str(y_train_i), file = file_result)



file_result.close()

plt.plot(range(n_epochs), losses)
plt.xlabel("epoch")
plt.ylabel("cross entropy loss")
plt.show()
