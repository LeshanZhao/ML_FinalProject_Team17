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
        act_input = np.dot(x, w)
        
        # Prevent the overflow error
        if act_input >= 500:
            self.output = 1
        elif act_input <= -500:
            self.output = 0
        else:    
            self.output = self.act_func()
        
        return self.output
    
    
    

        