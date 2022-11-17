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
    def pred(self, x, w):
        self.x_j = x
        self.output = self.act_func(np.dot(x, w))
        return self.output
    
    
    

        