# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:56:14 2022

@author: heckenna
"""
import perceptron
import numpy as np

class Layer:
    
    def __init__(self, perceptron_list, weights = None):
        self.perceptron_list = perceptron_list
        
        if weights != None:    
            self.weight_matrix = weights
        else:
            pass
        
    def predict(self, inpt):
        """
        # Gets output of hidden layer based on input inpt
        output = []
        
        for w_row, perc in zip(weight_matrix, perceptron_list):
            output.append(perc.predict(np.dot(w_row, inpt)))
            
        return output
        #"""
        return [perc.predict(np.dot(w_row, inpt)) for w_row, perc in zip(weight_matrix, perceptron_list)]        
        
        