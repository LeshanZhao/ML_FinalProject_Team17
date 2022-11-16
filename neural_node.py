# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:56:13 2022

@author: heckenna
"""

import numpy as np
import math


# Basically Perceptron Class


class Neural_Node:
    
    def __init__(self, size, 
                 func = lambda x: 1/(1 + math.exp(-x))): # Sigmoid by default
        self.size = size
        self.act_func = func

            
    # Gonna do dot product before this piece...
    def pred(self, x_dot_w):
        self.output = self.act_func(x_dot_w)
        return self.output
    
    #def sigmoid(self, row):
    #    return 1/(1 + math.exp(-np.dot(row, self.w_t)))
    
    def change_weights(self, w_t, delta_k):
        # TODO use gradient to compute new weights
        return w_t
    

        