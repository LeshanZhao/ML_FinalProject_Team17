# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:55:12 2022

@author: heckenna
"""
from perceptron import Perceptron
from layer import Layer
import numpy as np

class MLP:

    # TODO: Write the init function to take these parameters
    def __init__(self, 
                num_features, 
                num_hidden_layers, 
                hidden_sizes, 
                n_epochs = 1, 
                lr = 1,
                include_bias = False):
        # num_features is number of features we will have. Size of input layer
        # num_hidden_layers = number of hidden layers we will have
        # hidden_sizes = list of integers of length num_hidden_layers. 
        #     hidden_sizes[i] = size of hidden_layer i
        #     hidden_sizes[i - 1] = number of inputs for hidden_layer i
        # TODO: build the different layers here using the constructor in Layer class
        self.num_features = num_features
        self.lr = lr
        self.n_epochs = n_epochs
        self.losses = []
        
        bias = (1 if include_bias else 0)

        self.input_layer= Layer(num_features, num_features, is_input_layer = True, include_bias = include_bias)
        
        last_size = num_features + bias
        
        self.hidden_layers = []
        for j in range(num_hidden_layers):
            if j == 0:
                layer = Layer(hidden_sizes[j], last_size, include_bias = include_bias)
            else: 
                layer = Layer(hidden_sizes[j], last_size, include_bias = include_bias)  
            last_size = hidden_sizes[j] + bias
            self.hidden_layers.append(layer)
        
        self.output_layer = Layer(1, last_size) 
            

    def print_network(self):
        layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        #  + self.hidden_layers + list(self.output_layer)
        for i, layer in enumerate(layers):
            if (layer.is_input):
                print("\nInput Layer: ",end='\n')
            elif (len(layer.node_list) == 1):
                print("\nOutput Layer: ",end='\n')
            else:
                print("\nHidden Layer " + str(i) + ": ", end='\n')
            
            for j, node in enumerate (layer.node_list):
                print("\tNode_" + str(j) + ": ", end='')
                print("\tweights: " + str(layer.weight_matrix[j]))
                # print("\tOutput: " + str(1))
            
            # for weights in layer.weight_matrix:
            #     print(weights)


    def train(self, X, y, lr = None, batch_size = 25, epochs = None):
        if epochs is None:
            epochs = self.n_epochs
        [self.train_df(X, y, lr) for e in range(epochs)]
        
        
        """
        for i in range(int(X.shape[0]/batch_size)):
            # Runs training on batches of 25
            for e in range(epochs):
                for j in range(i*batch_size,(i+1)*batch_size):
                    row = X.iloc[j]
                    y_targ = y.iloc[j]
                    self.train_row(row, y_targ, lr)
        """
        
    def train_df(self, X, y, lr):
        # Uses lexical scoping and list comprehensions.
        # Right now, batch size unused. Still have overhead here, but better than before
        func_series = X.apply(lambda row: (lambda y: self.train_row(row, y, lr)), axis=1)
        losses = sum([func_series.iloc[i](y.iloc[i]) for i in range(len(y))])
        
        # After doing all the rows in the batch, alter weights...
        # Rather than alter weights every time.
        self.do_weight_changes()
        
        self.losses.append(losses)
    
    def train_row(self, row, y_targ, lr = None):
        loss = self._forward(row)
        self._backward(y_targ, lr)
        
        return loss
        # This line uncommented makes it do stochastic grad descent instead
        #self.do_weight_changes()
        
    def do_weight_changes(self):
        self.input_layer.do_weight_change()
        self.output_layer.do_weight_change()
        
        for layer in self.hidden_layers:
            layer.do_weight_change()

    def _forward(self,row):
        # get output from input layer
        y_last_layer = self.input_layer.forward(row)


        # hidden layers
        # TODO: Remove some overhead getting rid of for loop in some way 
        for nxt_hidden_layer in self.hidden_layers:
            y_last_layer = nxt_hidden_layer.forward(y_last_layer)
        
        y_output_layer_list = self.output_layer.forward(y_last_layer)

        
        output_result = y_output_layer_list[0]
        
        return output_result #TODO revert to returning 0 or 1. Returns now to help debug
        if output_result >= .5:
            return 1
        else:
            return 0
        
    
    # Does bacward prop for a given row/target
    def _backward(self, targ_y, lr = None):
        if lr is None:
            lr = self.lr

        delta_next_layer = self.output_layer.backward(lr, 
                                                      y_train = targ_y)
        next_hidden_layer = self.output_layer
        
        # TODO: Remove for loop overhead? 
        for i in range(len(self.hidden_layers)-1, -1, -1):
            
            current_hidden_layer = self.hidden_layers[i]
            delta_next_layer = current_hidden_layer.backward(lr, 
                                                        next_deltas = delta_next_layer,
                                                        next_weights = next_hidden_layer.weight_matrix)
            next_hidden_layer = current_hidden_layer
        
        self.input_layer.backward(lr,
                                  next_deltas = delta_next_layer,
                                  next_weights= next_hidden_layer.weight_matrix)

    
    # Predicts the target based on rows of input
    def pred(self, rows):
        #Y = []
        #for i in range(rows.shape[0]): 
        #    row = rows.iloc[i]
        #    y = self.pred_row(row)
        #    Y.append(y)
        return rows.apply(self.pred_row, axis=1)

    # Helper function. Predicts a single row
    def pred_row(self, row):
       return self._forward(row)
        

    # Loss function. This is what we want to minimize
    def loss(self, rows, targ):
        preds = self.pred(rows)
        
        #return sum((preds - targ)**2)
        #return sum(- targ*(np.log(preds)))

        # -[t*ln(y) + (1 - t)ln(1-y)]
        return sum(-(targ*np.log(preds) + (1-targ)*np.log(1 - preds)))
    
    
    
    