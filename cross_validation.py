# -*- coding: utf-8 -*-
"""
Created on Thu Nov  17 20:14:10 2022

@author: mza0200
"""
import numpy as np
import pandas as pd

class cross_validation: 
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.y = dataset.iloc[:,[-1]]
        self.X = dataset.drop(self.y,axis = 1)

        
    def train_test_split(self, features, targ, test_size=0.2, random_state=9208):
        #find the train percentage
        train_percentage = (1 - test_size) * 100
        test_percentage = test_size * 100
        #find num of features using the percentage of both test and train
        train_data_size = len(features) - (len(features) % train_percentage)
        test_data_size = len(features) - (len(features) % test_percentage)

        #split the features and labels into test part and train part
        X_train, y_train = features[:, train_data_size], targ[: , train_data_size]
        X_test, y_test =  features[:, test_data_size], targ[:,test_data_size]

        return X_train, X_test, y_train, y_test

    def cross_val_score(self, estimator , X , y , cv=0 ): 
        folds = self.Kfold(cv)
        conf_matrix = [cv] 
        for fold in folds: 
            conf_matrix.append(self.confusion_matrix())
        return

    def createFold(self, start_index, fold_size):
        return self.dataset.iloc[start_index: fold_size,:]

    def Kfold (self, n_splits):
        # Use n_splits to consider the number of folds 
        # Create each fold from the data starting from index i*fold_size until (i * fold_size + fold_size)
        # For Ex: fold_size is 50 
        # The 1st Fold will start from 0*50 until (0 * 50 + 50) in data 
        # The 2nd Fold from 1 * 50 until (1 * 50 + 50 )  
           
        fold_size = len(self.dataset)/n_splits
        folds = [] 
        for i in n_splits:
            folds.append(self.createFold(i*fold_size, fold_size))
        return folds


    def confusion_matrix(self, y_test, y_pred): #done!!
        TruePostive = []
        FalsePostive = []
        TrueNegative = []
        FalseNegative = []
        for y, y_ in y_test, y_pred: 
            if y == 1 and y_==1:
                TruePostive.append(y)
            elif y == 0 and y_ == 0:
                FalsePostive.append(y)
            elif y == 1 and y_ == 0:
                TrueNegative.append(y)
            elif y == 0 and y_ == 1:
                FalseNegative.append(y)
        
        TP = len(TruePostive)
        FP = len(FalsePostive)
        TN = len(TrueNegative)
        FN = len(FalseNegative)

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        return Precision, Recall, Accuracy
  