# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:56:14 2022

@author: heckenna
"""
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
        self.prev_weight_change = 0# No longer used because we stopped using momentum
        self.alpha = .25 # No longer used because we stopped using momentum
        bias = (1 if self.bias else 0)
        
        
        self.node_list = self._build_percepton_list(num_perceptrons + bias, num_inputs)

        if self.is_input:
            # input is funky...
            self.weight_matrix = self._build_weight_matrix(num_perceptrons + bias, 1)
        else:
            # Initialize weights on U(-num_perc, num_perc). 
            self.weight_matrix = self._build_weight_matrix(num_perceptrons + bias, num_inputs)
        
        self.weight_changes = np.zeros(self.weight_matrix.shape)
        
    # Does foward direction of mlp.
    def forward(self, layer_input):
        # # Gets output of hidden layer based on layer_input
        # self.layer_input = layer_input
                
        if self.is_input:
            # We just have 1 input per node...
            # Weights are list not matrix
            """
            for i in range(len(self.node_list)):
                w_i = self.weight_matrix[i] # w_i is List of length 1
                perc  = self.node_list[i]
                
                # Putting these a 1 element lists for sake of np.dot
                output.append(perc.pred(w_i, [layer_input[i]]))
            """
            # List comprehension for speed            
            layer_input = [np.array([lay])for lay in layer_input]
            
            if self.bias:
                layer_input.insert(0, np.array([0]))
            return [self.node_list[i].pred(x = layer_input[i], w = self.weight_matrix[i]) for i in range(len(self.node_list))]
        
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
        return [self.node_list[i].pred(x = layer_input, w = self.weight_matrix[i]) for i in range(len(self.node_list))]
            
        
        #return [perc.predict(np.dot(w_row, layer_input)) for w_row, perc in zip(weight_matrix, perceptron_list)]        
        

    
    # y_train provided only in output layer. next_delta = None only if output
    def backward(self, lr, next_deltas = None, next_weights = None, y_train = None):
        ## If output layer... mul_term = (y_train - o_k)
        ## Else: mul_term = sum([next_weights[i][h]*delta[h] for i in range(len(delta))]) 
        # TODO: Remove overhead. This is most inefficient function as far as I can tell
        
        deltas = self._compute_deltas(next_deltas, next_weights, y_train)
        
        
        if self.is_input:
            """
            for i in range(len(self.node_list)):
                node = self.node_list[i]
                delt = deltas[i]
                feature = node.x_j[0]
                
                w_change = lr*np.array(delt)*feature #+ self.prev_weight_change*self.alpha
                
                self.weight_changes[i] = w_change + self.weight_changes[i]
            """
            # List comprehension for speed. Leaving for loop as comment for clarity
            self.weight_changes = [self.weight_changes[i] + lr*np.array(deltas[i])*(self.node_list[i].x_j[0]) for i in range(len(self.node_list))]
            return deltas # No need to return anything... input layer. Will anyways
        """
        for node, delt, weight_list in zip(self.node_list, deltas, self.weight_matrix):
            x_j = node.x_j 
            w_change = lr*np.array(delt)*np.array(x_j)
            
            # This should change weight matrix via mutation 
            for i in range(len(weight_list)):
                if abs(weight_list[i] + w_change[i]) >= weight_max:
                    continue
                weight_list[i] += w_change[i]
        """
        [self._alter_weights_non_input(lr, self.node_list[i], deltas[i], self.weight_matrix[i]) for i in range(len(deltas))]
        # Returns deltas so they can be used for earlier layer back prop
        return deltas
    
    # Helper function for backward prop
    # Calling it for its side effects.
    # Probably should rename. At this point, it doesn't actually alter weights
    #    It saves what the weight changes should be for later when we want to 
    #    do the changes
    def _alter_weights_non_input(self, lr, node, delt, weight_list):
        
        x_j = node.x_j 
        w_change = lr*np.array(delt)*np.array(x_j) #+ self.prev_weight_change*self.alpha
        self.weight_changes += np.array(w_change)
        #self.prev_weight_change = w_change
        '''
        # This will change weight matrix via mutation. Slow though
        for i in range(len(weight_list)):
            #if abs(weight_list[i] + w_change[i]) >= weight_max:
            #    continue
            weight_list[i] += w_change[i]
        '''
            
    # TODO: I am making it determine the weight changes for multiple inputs
    # then changing the list of weights afterwards.
    def _do_weight_change(self, size = 1):
        w_delt = list(map(lambda x: x/size, self.weight_changes))
        self.weight_matrix = self.weight_matrix + w_delt
        self.weight_changes = np.zeros(self.weight_matrix.shape)
    
    def _compute_deltas(self, next_deltas = None, next_weights = None, y_train = None):
        # Output layer case
        if not(y_train is None):
            node = self.node_list[0]
            o_k = node.output # Sigmoid output
            
            # Determined by derivated of loss function
            #mul_term = ((y_train)/o_k - (1-y_train)/(1 - o_k))
            
            mul_term = ((y_train)/o_k - (1-y_train)/(1-o_k))

            #if o_k >= .5:
            #    mul_term = (y_train - 1)
            #else:
            #    mul_term = y_train

            
            d_k = self._compute_delta_helper(o_k, mul_term)
            
            # returning list of deltas
            return [d_k]
            
        return [self._compute_delta_helper_vec(h, next_weights, next_deltas) for h in range(len(self.node_list))]
        
        
    def _compute_delta_helper_vec(self, h, next_weights, next_deltas):
        node = self.node_list[h]
        #mul_term = sum([next_weights[k][h] * next_deltas[k] for k in range(len(next_deltas))])
        mul_term = np.dot(np.transpose(next_weights)[h], next_deltas)
        o_k = node.output
        
        # Return d_k
        return self._compute_delta_helper(o_k, mul_term)
            
    def _compute_delta_helper(self, o_k, mul_term):
        # Computes delta of the node.
        
        delta_k = o_k * (1 - o_k) * mul_term
        return delta_k
    
    # Used in init to create the list of perceptrons in the layer.
    def _build_percepton_list(self, num_perceptrons, num_inputs):
        build_perc = (lambda i: neural_node.Neural_Node(size = num_inputs, func = lambda y: 1) if (i == 0 and self.bias) else neural_node.Neural_Node(size = num_inputs))
        return [build_perc(i) for i in range(num_perceptrons)]
        
    # Builds weight matrix. 
    # These weights go into the perceptrons of the perceptron list
    def _build_weight_matrix(self, num_perceptrons, num_inputs):
        weights = []
        
        for i in range(num_perceptrons):
            p_weights = []
            for j in range(num_inputs):
                p_weights.append(random.randrange(-num_perceptrons, num_perceptrons + 1))
            weights.append(p_weights)
            
        return np.array(weights)

