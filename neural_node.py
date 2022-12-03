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
                 func = lambda x: (1/(1 + math.exp(-x))) if abs(x) < 500 else (lambda y: .999 if y>500 else .001)(x)): # Sigmoid by default. If statement part removes overflow possibilities
        self.size = size
        self.act_func = func

            
    # Gonna x and w need to be np arrays
    def pred(self, x = 1, w = 1): # Setting defaults so order doesnt get goofed up on accident
        self.x_j = x
        
        
        # I think I avoided the overflows a different way
        act_input = np.dot(x, w)
        
        # Prevent the overflow error
        #if act_input >= 500:
        #    self.output = .999
        #elif act_input <= -500:
        #    self.output = 0.001
        #else:    
        self.output = self.act_func(act_input)
        #"""
        
        #self.output = self.act_func(np.dot(x, w))
        return self.output
    
    
    

        