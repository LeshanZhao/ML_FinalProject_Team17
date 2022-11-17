# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:55:12 2022

@author: heckenna
"""
from perceptron import Perceptron
from layer import Layer

class MLP:
    def __init__(self, 
                Input_Layer: Layer, 
                Output_Layer: Layer,
                Hidden_Layers: list[Layer],
                X,
                y):
        self.input_layer = Input_Layer
        self.output_layer = Output_Layer
        self.hidden_layers = Hidden_Layers
        self.X = X # TODO no need to initialize X and Y
        self.y = y

    def train(self, X, y):
        for row, y_targ in zip(X, y):
            self.train_row(row, y_targ)

    def train_row(self, row, y_targ):
        self._forward(row)
        self._backward(row, y_targ)
        return

    def _forward(self, x):
        # get output from input layer
        y_last_layer = self.input_layer.forward(self.X)

        # hidden layers
        print("forward start")
        for nxt_hidden_layer in self.hidden_layers:
            y_last_layer = nxt_hidden_layer.forward(y_last_layer)

        y_output_layer_list = self.output_layer.forward(y_last_layer)

        print("forward done")
        
        output_result = y_output_layer_list[0]
        
        if output_result >= .5:
            return 1
        return 0
        
    
    def _backward(self):
        print("Backward start")
        # TODO
        
        y_output_layer_list = self.output_layer.forward(y_last_layer)
        
        for nxt_hidden_layer in self.hidden_layers:
            y_last_layer = nxt_hidden_layer.forward(y_last_layer)
        
        print("Backward done (nothing done yet)")
        




    def pred(self, row):

        return 
    
    