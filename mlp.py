# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:55:12 2022

@author: heckenna
"""
from perceptron import Perceptron
from layer import Layer

class MLP:
    def dep__init__(self, 
                Input_Layer, 
                Output_Layer,
                Hidden_Layers,
                X,
                y):
        # TODO: instead of passing in layers, tell it how many layers and what sizes
        # We should build the layers here
        
        self.input_layer = Input_Layer
        self.output_layer = Output_Layer
        self.hidden_layers = Hidden_Layers
        self.X = X # TODO no need to initialize X and Y
        self.y = y

    # TODO: Write the init function to take these parameters
    def __init__(self, 
                num_features, 
                num_hidden_layers, 
                hidden_sizes, 
                n_epochs = 1, 
                lr = 1):
        # num_features is number of features we will have. Size of input layer
        # num_hidden_layers = number of hidden layers we will have
        # hidden_sizes = list of integers of length num_hidden_layers. 
        #     hidden_sizes[i] = size of hidden_layer i
        #     hidden_sizes[i - 1] = number of inputs for hidden_layer i
        # TODO: build the different layers here using the constructor in Layer class
        self.num_features = num_features
        self.lr = lr
        self.n_epochs = n_epochs

        self.input_layer= Layer(num_features, num_features, is_input_layer = True)

        self.hidden_layers = []
        for j in range(num_hidden_layers):
            if j == 0:
                layer = Layer(hidden_sizes[j], self.num_features)
            else: 
                layer = Layer(hidden_sizes[j], hidden_sizes[j-1])  
            self.hidden_layers.append(layer)
        
        self.output_layer = Layer(1, hidden_sizes[-1]) 
            

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


    def train(self, X, y, lr = None):
        for row, y_targ in zip(X, y):
            self.train_row(row, y_targ, lr)

    def train_row(self, row, y_targ, lr = None):
        self._forward(row)
        self._backward(row, y_targ, lr)

    def _forward(self,row):
        # get output from input layer
        y_last_layer = self.input_layer.forward(row)

        # hidden layers
        print("forward start")
        for nxt_hidden_layer in self.hidden_layers:
            y_last_layer = nxt_hidden_layer.forward(y_last_layer)

        y_output_layer_list = self.output_layer.forward(y_last_layer)

        print("forward done")
        
        output_result = y_output_layer_list[0]
        
        if output_result >= .5:
            return 1
        else:
            return 0
        
    
    def _backward(self, targ_y, lr = None):
        if lr is None:
            lr = self.lr
        print("Backward start")
        # TODO
        delta_next_layer = [self.output_layer.backward(self.lr, 
                                                       y_train = targ_y)]
        
        for i in range(len(self.hidden_layers), -1, -1):
            next_hidden_layer = self.hidden_layers[i]
            delta_next_layer = next_hidden_layer.backward(self.lr, 
                                                        next_deltas = delta_next_layer,
                                                        next_weights = next_hidden_layer.weight_matrix)
        self.input_layer.backward(self.lr,
                                  next_deltas = delta_next_layer,
                                  next_weights= next_hidden_layer.weight_matrix)

        print("Backward done (nothing done yet)")
        

    def pred(self, rows):
        Y = []
        for row in rows: 
            y = self.pred_row(row)
            Y.append(y)
        return Y

    def pred_row(self, row):
       return self._forward(row)
        

    
    
    
    