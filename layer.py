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
    
    def __init__(self, num_perceptrons, num_inputs, include_bias = False, is_input_layer = False):
        # num_perceptrons = number of perceptrons in the layer
        # num_inputs = size of the input vector to each node of the layer.
        #     is num_perceptrons for previous layer
        # is_input_layer = True if is input layer, otherwise false
        self.bias = include_bias
        self.is_input = is_input_layer
        self.node_list = self.build_percepton_list(num_perceptrons, num_inputs)
        
        if self.is_input:
            # input is funky...
            self.weight_matrix = self.build_weight_matrix(num_perceptrons, 1)
        else:
            # Initialize weights on U(-num_perc, num_perc). 
            self.weight_matrix = self.build_weight_matrix(num_perceptrons, num_inputs)
            
        
    def forward(self, layer_input):
        # Gets output of hidden layer based on layer_input
        #self.layer_input = layer_input
        
        """
        # If we add bias, we do it another way...
        output = []
        
        if self.bias:
            output.append(1)
            
        """
        
        if self.is_input:
            # We just have 1 input per node...
            # Weights are list not matrix
            """
            for i in range(len(self.node_list)):
                w_i = self.weight_matrix[i] # w_i is List of length 1
                perc  = self.node_list[i]
                
                # Putting these a 1 element lists for sake of np.dot
                output.append(perc.pred(w_i, [layer_input[i]])) #Ternary???
            """
            # List comprehension. Speed
            layer_input = [np.array([lay])for lay in layer_input]
            return [self.node_list[i].pred(self.weight_matrix[i], layer_input[i]) for i in range(len(self.node_list))]
        
        # If we are not input...
        
        """
        # Has each percepton do a prediction
        for i in range(len(self.node_list)):
            w_row = self.weight_matrix[i]
            perc  = self.node_list[i]
            
            # For node node_list[i], weight_matrix[i] is corresponding weights
            output.append(perc.pred(layer_input, w_row))
        """
        # List comprehension is faster
        return [self.node_list[i].pred(layer_input, self.weight_matrix[i]) for i in range(len(self.node_list))]
            
        
        #return [perc.predict(np.dot(w_row, layer_input)) for w_row, perc in zip(weight_matrix, perceptron_list)]        
        

    
    # y_train provided only in output layer. delta None if output
    def backward(self, lr, next_deltas = None, next_weights = None, y_train = None):
        ## If output layer... mul_term = (y_train - o_k)
        ## Else: mul_term = sum([next_weights[i][h]*delta[h] for i in range(len(delta))]) 
        # TODO: Remove overhead. This is most inefficient function as far as I can tell
        weight_max = 13
        
        deltas = self.compute_deltas(next_deltas, next_weights, y_train)
        
        
        if self.is_input:
            for i in range(len(self.node_list)):
                node = self.node_list[i]
                delt = deltas[i]
                feature = node.x_j[0]
                
                w_change = lr*np.array(delt)*feature
                
                if abs(self.weight_matrix[i][0] + w_change) >= weight_max:
                    continue
                self.weight_matrix[i][0] += w_change
            return deltas # No need to return anything... input layer
        
        for node, delt, weight_list in zip(self.node_list, deltas, self.weight_matrix):
            x_j = node.x_j 
            w_change = lr*np.array(delt)*np.array(x_j)
            
            # This should change weight matrix via mutation 
            for i in range(len(weight_list)):
                if abs(weight_list[i] + w_change[i]) >= weight_max:
                    continue
                weight_list[i] += w_change[i]
        
        # Returns deltas so they can be used for earlier layer back prop
        return deltas
        
    def compute_deltas(self, next_deltas = None, next_weights = None, y_train = None):
        # Output layer case
        if not(y_train is None):
            node = self.node_list[0]
            o_k = node.output # Needs to be threshold?
            #if ok >=.5:
            #    o_k = 1
            #else:
            #    o_k = 0
            
            # Going to use threshold only here to see if progress...
            mul_term = (y_train - o_k)

            if (o_k >= 0.75 and y_train == 1) or (o_k < .25 and y_train == 0):
                mul_term = mul_term
            #else:
            #    mul_term = 1

            
            d_k = self.compute_delta_helper(node, mul_term)
            
            # returning list of deltas
            return [d_k]
            
        """
        # Otherwise we are input or hidden layer. These nodes need future info...
        d_k_list = []
        
        for h in range(len(self.node_list)):
            node = self.node_list[h]
            mul_term = sum([next_weights[k][h] * next_deltas[k] for k in range(len(next_deltas))])
            
            d_k = self.compute_delta_helper(node, mul_term)
            
            # TODO remove append for efficiency
            d_k_list.append(d_k)
        return d_k_list
        """
        
        return [self.compute_delta_helper_vec(h, next_weights, next_deltas) for h in range(len(self.node_list))]
        
        
    def compute_delta_helper_vec(self, h, next_weights, next_deltas):
        node = self.node_list[h]
        mul_term = sum([next_weights[k][h] * next_deltas[k] for k in range(len(next_deltas))])
            
        # Return d_k
        return self.compute_delta_helper(node, mul_term)
            
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
            
        return weights
            
