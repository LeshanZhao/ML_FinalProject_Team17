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
        self.X = X
        self.y = y


    def train(self):
        self._forward()
        self._backward()
        return

    def _forward(self):
        # get output from input layer
        y_last_layer = self.input_layer.predict(self.X)

        # hidden layers
        print("forward start")
        for nxt_hidden_layer in self.hidden_layers:
            y_last_layer = nxt_hidden_layer.predict(y_last_layer)

        y_output_layer_list = self.output_layer.predict(y_last_layer)

        print("forward done")
        
        output_result = y_output_layer_list[0]
        
        if output_result >= .5:
            return 1
        return 0
        
    
    def _backward(self):
        print("Backward start")
        # TODO
        
        print("Backward done (nothing done yet)")
        




    def pred(self, row):

        return 
    
    