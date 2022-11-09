# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 18:36:36 2022

@author: heckenna
"""
import numpy as np


# Perceptron Class


class Perceptron:
    
    def __init__(self, size, weight = 1, w_vec = None, w_init_val = 0, epochs = 1):
        self.epochs = epochs
        self.weight = weight
        self.size = size
        
        if w_vec == None:
            self.w_t = (np.zeros(size) + w_init_val)
        else:
            self.w_t = w_vec
            
    def pred_row(self, row):
        if np.dot(self.w_t, row) >= 0:
            return 1
        return 0
    
    def train_row(self, row, targ):
        self.w_t += (targ - self.pred_row(row))*row
        
    def train_classifier(self, train_df, target_col):
        
        for e in range(self.epochs):
            for i in range(train_df.shape[0]):
                self.train_row(train_df.iloc[i], target_col.iloc[i])
       
        
        return
    
    def predict_data(self, data):
        print(self.w_t)
        return [self.pred_row(data.iloc[i]) for i in range(data.shape[0])]
        
    
    
        
    
    
    
    