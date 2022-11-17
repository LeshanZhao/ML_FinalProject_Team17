# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:56:14 2022

@author: heckenna
"""
import perceptron
import neural_node
import numpy as np
import random

class Layer:
    
    def __init__(self, num_perceptrons, num_inputs, include_bias = False, weights = None):
        self.bias = include_bias
        
        self.node_list = self.build_percepton_list(num_perceptrons, num_inputs)
        
        if not (weights is None):
            self.weight_matrix = weights
        else:
            # Initialize weights on U(-num_perc, num_perc). 
            self.weight_matrix = self.build_weight_matrix(num_perceptrons, num_inputs)
            
        
    def forward(self, layer_input):
        # Gets output of hidden layer based on layer_input
        #self.layer_input = layer_input
        output = []
        
        if self.bias:
            output.append(1)
        
        # Has each percepton do a prediction
        for w_row, perc in zip(self.weight_matrix, self.node_list):
            # For node node_list[i], weight_matrix[i] is corresponding weights
            output.append(perc.pred(w_row, layer_input))
            
        return output
        
        #return [perc.predict(np.dot(w_row, layer_input)) for w_row, perc in zip(weight_matrix, perceptron_list)]        
        

    
    # y_train provided only in output layer. delta None if output
    def backward(self, lr, next_deltas = None, next_weights = None, y_train = None):
        ## If output layer... mul_term = (y_train - o_k)
        ## Else: mul_term = sum([next_weights[i][h]*delta[h] for i in range(len(delta))]) 
        # TODO
        deltas = compute_deltas(next_deltas, next_weights, y_train)
        
        for node, delt, weight_list in zip(self.nodes, deltas, self.weight_matrix):
            x_j = node.x_j 
            w_change = lr*delt*x_j
            
            # I think this should change weight matrix via mutation... 
            # Can come back later to confirm
            for i in range(len(weight_list)):
                weight_list[i] += w_change[i]
            
            
            
        
        # Returns deltas so they can be used for earlier layer back prop
        return deltas
        
    def compute_deltas(self, next_deltas = None, next_weights = None, y_train = None):
        # Output layer case
        if (y_train != None):
            node = self.node_list[0]
            o_k = node.output # Needs to be threshold?
            
            mul_term = (y_train - o_k)
            
            d_k = self.compute_delta_helper(node, mul_term)
            
            # returning list of deltas
            return [d_k]
            
        # Otherwise we are input or hidden layer. These nodes need future info...
        d_k_list = []
        
        for h in range(len(self.node_list)):
            node = self.node_list[h]
            mul_term = sum([next_weights[h][k] * next_deltas[k] for k in range(len(next_deltas))])
            
            d_k = self.compute_delta_helper(node, mul_term)
            d_k_list.append(d_k)
        
   
        return d_k_list
    
    def compute_delta_helper(self, node, mul_term):
        # Computes delta of the node.
        o_k = node.output
        
        delta_k = o_k * (1 - o_k) * mul_term
        return delta_k
    
    def build_percepton_list(self, num_perceptrons, num_inputs):
        return [neural_node.Neural_Node(size = num_inputs) for i in range(num_perceptrons)]
        
    def build_weight_matrix(self, num_perceptrons, num_inputs):
        weights = []
        
        for i in range(num_perceptrons):
            p_weights = []
            for j in range(num_inputs):
                p_weights.append(random.randrange(-num_perceptrons, num_perceptrons + 1))
            weights.append(p_weights)
            
