# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:56:14 2022

@author: heckenna
"""
import perceptron
import numpy as np
import random

class Layer:
    
    def __init__(self, num_perceptrons, num_inputs, include_bias = False, weights = None):
        self.bias = include_bias
        
        self.perceptron_list = self.build_percepton_list(num_perceptrons, num_inputs)
        
        if weights != None:    
            self.weight_matrix = weights
        else:
            # Initialize weights on U(0,1). Maybe do differently
            self.weight_matrix = self.build_weight_matrix(num_perceptrons, num_inputs)
            
        
    def predict(self, layer_input):
        # Gets output of hidden layer based on layer_input
        output = []
        
        if self.bias:
            output.append(1)
        
        # Has each percepton do a prediction
        for w_row, perc in zip(self.weight_matrix, self.perceptron_list):
            output.append(perc.predict(np.dot(w_row, layer_input)))
            
        return output
        
        #return [perc.predict(np.dot(w_row, layer_input)) for w_row, perc in zip(weight_matrix, perceptron_list)]        
        
    def build_percepton_list(self, num_perceptrons, num_inputs):
        return [perceptron.Perceptron(size = num_inputs) for i in range(num_perceptrons)]
        
    def build_weight_matrix(self, num_perceptrons, num_inputs):
        weights = []
        
        for i in range(num_perceptrons):
            p_weights = []
            for j in range(num_inputs):
                p_weights.append(random.randrange(-num_perceptrons, num_perceptrons + 1))
            weights.append(p_weights)
            
