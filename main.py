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

cv = cross_validation()


def get_acc(pred, y):
    return (1 - np.sum(abs(np.array(y) - np.array(pred))) / len(y)) * 100


# random.seed(12321)

data = data_opener.get_data()
# Max corr is .5, so correlation not going to be used for feature selection
corr_matrix = data.corr()

X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)


n_samples = 10000

X_train = X_train.head(n_samples)
y_train = y_train.head(n_samples)
# print(X_train.iloc[0])

n_features = 13
lr = .00001
n_epochs = 200
batch_size = 25

output_path = "output/" + \
                "lr" + str(lr) + "_" + \
                "epochs" + str(n_epochs) + "_" + \
                "samples" + str(n_samples)
                # "batchSize" + str(batch_size) + "_" + \
result_txt = output_path + ".txt"
# result_txt_train = output_path + "_train.txt"
# result_txt_test = output_path + "_test.txt"
result_png_losscurve = output_path + ".png"

# train_result_file = open(result_txt, "a+")
# test_result_file = open(result_txt_test, "a+")
result_file = open(result_txt, "a+")

print("\n\n=============== Start training... ===============\n", file = result_file)

print("sample size: ", n_samples, file = result_file)
print("lr: ", lr, file = result_file)
print("n_epochs: ", n_epochs, file = result_file)
print("batch_size: ", batch_size, file = result_file)

mlp_clf = mlp.MLP(n_features = n_features, 
                    hidden_sizes = [8], 
                    n_epochs = n_epochs,
                    batch_size = batch_size,
                    include_bias = True)

mlp_clf.train(X_train, y_train, lr = lr)
print("\n=============== Training done. ===============\n\n", file = result_file)

# print cross entropy loss values after each epoch
losses = mlp_clf.losses
print("losses: ", losses, file = result_file)
print()

plt.plot(range(n_epochs), losses)
plt.xlabel("epoch")
plt.ylabel("cross entropy loss")
plt.savefig(result_png_losscurve)

print("\n=============== Predict on training data ===============\n", file = result_file)
# Predict on Training set
y_pred = mlp_clf.pred(X_train)
pred_sigmoid_train = y_pred
y_pred_train = list(map(lambda x: 1 if x >= .5 else 0, y_pred))

# print("Accuracy:", get_acc(y_pred_train, y_train), file = result_txt)

# print statistics and confusion matrix
stats_train = cv.print_stat(y_train, y_pred_train)
print(stats_train[0], file = result_file)
print(stats_train[1], file = result_file)



print("\n=============== Predict on testing data ===============\n", file = result_file)
# Predict on Training set
y_pred = mlp_clf.pred(X_test)
pred_sigmoid_test = y_pred
y_pred_test = list(map(lambda x: 1 if x >= .5 else 0, y_pred))

# print statistics and confusion matrix
stats_test = cv.print_stat(y_test, y_pred_test)
print(stats_test[0], file = result_file)
print(stats_test[1], file = result_file)




print("\n\n=============== Sigmoid output on training data ===============\n", file = result_file)

# print output from sigmoid and the corresponding true label for training set
for sigmoid_i, y_train_i in zip(pred_sigmoid_train, y_train):
    print(str(sigmoid_i), "", str(y_train_i), file = result_file)
print("\* ==============\n" + " * \n" * 20 + " * ============== *\\ \n", file = result_file)


print("\n\n=============== Sigmoid output on testing data ===============\n", file = result_file)

# print output from sigmoid and the corresponding true label for test set
for sigmoid_i, y_test_i in zip(pred_sigmoid_test, y_test):
    print(str(sigmoid_i), "", str(y_test_i), file = result_file)
print("\* ==============\n" + " * \n" * 20 + " * ============== *\\ \n", file = result_file)


result_file.close()
