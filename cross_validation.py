# -*- coding: utf-8 -*-
"""
Created on Thu Nov  17 20:14:10 2022

@author: mza0200
"""
import numpy as np
import pandas as pd

class cross_validation: 
    def __init__(self, dataset = None) -> None:
        #split the dataset columns to features and label column
        if not dataset is None:
            self.dataset = dataset
            self.y = dataset.iloc[:,[-1]]
            self.X = dataset.drop(self.y,axis = 1)
        
        self.precision = 0
        self.recall = 0 
        self.accuracy = 0
        self.error = 0 

    def train_test_split(self, features, targ, test_size=0.2):
        #find the train percentage
        train_percentage = (1 - test_size) * 100
        test_percentage = test_size * 100
        #find num of features using the percentage of both test and train
        train_data_size = len(features) - (len(features) * train_percentage)
        test_data_size = len(features) - (len(features) * test_percentage)

        #split the features and labels into test part and train part
        X_train, y_train = features[:train_data_size, :], targ[: train_data_size, :]
        X_test, y_test =  features[: test_data_size,:], targ[: test_data_size,:]

        return X_train, X_test, y_train, y_test
   

    def createFold(self, start_index, fold_size):
        lastIndex = start_index + fold_size
        return self.dataset.iloc[int(start_index):int(lastIndex), :] #slice the fold from the start index to fold size bu added the fold size ti index to see the last index in fold

    def Kfold(self, n_splits, shuffle = True):
        # Use n_splits to consider the number of folds 
        # Create each fold from the data starting from index i*fold_size until (i * fold_size + fold_size)
        # For Ex: fold_size is 50 
        # The 1st Fold will start from 0*50 until (0 * 50 + 50) in data 
        # The 2nd Fold from 1 * 50 until (1 * 50 + 50 )  
        fold_size = len(self.dataset)/n_splits
        folds = [] 
        for i in range(n_splits):
            if shuffle == True: 
                self.dataset = self.dataset.sample(frac = 1)
            folds.append(self.createFold(i*fold_size, fold_size))
        return folds


    def confusion_matrix(self, y_test, y_pred): #done!!
        TruePostive = []
        FalsePostive = []
        TrueNegative = []
        FalseNegative = []
        for y, y_ in zip(y_test, y_pred): 
            if y == 1 and y_==1:
                TruePostive.append(y)
            elif y == 1 and y_ == 0:
                # FalsePostive.append(y)  # should be FN!
                FalseNegative.append(y)
            elif y == 0 and y_ == 0:
                TrueNegative.append(y)
            elif y == 0 and y_ == 1:
                # FalseNegative.append(y)  # should be FP!
                FalsePostive.append(y)  # should be FP!
        
        TP = len(TruePostive)
        FP = len(FalsePostive)
        TN = len(TrueNegative)
        FN = len(FalseNegative)
        
        size = len(y_test)
        
        return TP/size, FP/size, FN/size, TN/size

    def add_metrics(self, fold_nums, y_pred, y_test):
        TruePostive = []
        FalsePostive = []
        TrueNegative = []
        FalseNegative = []
        for y, y_ in zip(y_test, y_pred): 
            if y == 1 and y_==1:
                TruePostive.append(y)
            elif y == 1 and y_ == 0:
                # FalsePostive.append(y)  # should be FN!
                FalseNegative.append(y)
            elif y == 0 and y_ == 0:
                TrueNegative.append(y)
            elif y == 0 and y_ == 1:
                # FalseNegative.append(y)  # should be FP!
                FalsePostive.append(y)  # should be FP!
        
        TP = len(TruePostive)
        FP = len(FalsePostive)
        TN = len(TrueNegative)
        FN = len(FalseNegative)
        
        matrix = "\t\t" + "Positive\t" + "Negative\t\n" + \
            "pred_pos\t" + str(TP) + "\t\t" + str(FP) + "\t\n" \
            "pred_neg\t" + str(FN) + "\t\t" + str(TN) + "\t\n"
        # print("\t\t", "Positive\t", "Negative\t\n")
        # print("pred_posi\t", TP, "\t\t", FP, "\t\n")
        # print("pred_nega\t", FN, "\t\t", TN, "\t\n")

        size = len(y_test)
        
        if TP + FP == 0:
            Precision = 0
        else:
            Precision = TP / (TP + FP)
            
        if (TP + FN) == 0:
            Recall = 0
        else:
            Recall = TP / (TP + FN)
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        ErrorRate = 1 - Accuracy
        

        self.precision += Precision/fold_nums
        self.recall += Recall/fold_nums
        self.accuracy += Accuracy/fold_nums
        self.error += ErrorRate/fold_nums
        
        
    def cross_val_score(self, cv=2, shuffle=True): 
        if shuffle == True: 
            folds = self.Kfold(cv, shuffle)
        conf_matrix = [] 
        for fold in folds:
            conf_matrix.append(self.confusion_matrix())
        return conf_matrix

    def print_stat(self, y_test, y_pred): #done!!
        TruePostive = []
        FalsePostive = []
        TrueNegative = []
        FalseNegative = []
        for y, y_ in zip(y_test, y_pred): 
            if y == 1 and y_==1:
                TruePostive.append(y)
            elif y == 1 and y_ == 0:
                FalseNegative.append(y)  # should be FN!
            elif y == 0 and y_ == 0:
                TrueNegative.append(y)
            elif y == 0 and y_ == 1:
                FalsePostive.append(y)  # should be FP!
        
        TP = len(TruePostive)
        FP = len(FalsePostive)
        TN = len(TrueNegative)
        FN = len(FalseNegative)
        
        Precision = TP / (TP + FP) if (TP + FP != 0) else 0
        Recall = TP / (TP + FN) if (TP + FN != 0) else 0
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        ErrorRate = 1 - Accuracy

        metrics = "Accuracy:\t" + str(Accuracy) + "\n" + \
            "ErrorRate:\t" + str(ErrorRate) + "\n" + \
            "Precision:\t" + str(Precision) + "\n" + \
            "Recall:\t" + str(Recall) + "\n"


        Matrix = "Confusion matrix:\n" + \
            "\t\t\t" + "Positive\t" + "Negative\t\n" + \
            "pred_pos\t" + str(TP) + "\t\t\t" + str(FP) + "\t\n" \
            "pred_neg\t" + str(FN) + "\t\t\t" + str(TN) + "\t\n"

        # print("\t\t", "Positive\t", "Negative\t\n")
        # print("pred_posi\t", TP, "\t\t", FP, "\t\n")
        # print("pred_nega\t", FN, "\t\t", TN, "\t\n")
        return [metrics, Matrix]