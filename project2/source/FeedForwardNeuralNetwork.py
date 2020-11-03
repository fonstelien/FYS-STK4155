'''Implementation of the FFNN class (Feed-Forward Neural Network)'''

import numpy as np
import sklearn as skl
from StochasticGradientDescent import SGD

class Layer:
    
    ## Activation functions and their derivatives
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s*(1-s)
    
    _activation_funcs = {'sigmoid':_sigmoid}
    _activation_func_derivatives = {'sigmoid':_sigmoid_derivative}
    
    def __init__(self, num_nodes, num_inputs, activation_function='sigmoid'):
        np.random.seed(0)
        self.num_nodes = num_nodes
        self.biases = np.zeros(num_nodes) + .01
        self.weights = np.random.rand(num_inputs, num_nodes)
        self.activation_function = activation_function
        self.activation_inputs = None  # matrix (num_samples x num_nodes)
        self.outputs = None  # matrix (num_samples x num_nodes)
        self.deltas = None  # matrix (num_samples x num_nodes)
        
    def feed_forward(self, inputs):
        activation_func = self._activation_funcs[self.activation_function]
        self.activation_inputs = inputs @ self.weights + self.biases
        self.outputs = activation_func(self, self.activation_inputs)
        return self.outputs

    def back_propagate(self, cost_derivatives):
        activation_func_derivative = self._activation_func_derivatives[self.activation_function]
        activation_derivatives = activation_func_derivative(self, self.activation_inputs)
        self.deltas = cost_derivatives * activation_derivatives
        return (self.deltas, self.weights)
    
    def update_weights(self, eta, inputs):
        self.weights = self.weights - eta*(inputs.T @ self.deltas)
        return self.weights

    def update_biases(self, eta):
        self.biases = self.biases - eta*np.sum(self.deltas, axis=0)
        return self.biases
        
class FFNN:
    ## Cost function derivatives
    def _mse_derivative(self, A, T):
        return -2*(T - A)

    _cost_func_derivatives = {'mse':_mse_derivative}
    
    def __init__(self, num_features, cost_function='mse'):
        self.num_features = num_features
        self.cost_function = cost_function
        self.layers = list()

    def add_layer(self, num_nodes, activation_function='sigmoid'):
        num_inputs = self.num_features
        if (len(self.layers) > 0):
            preceding_layer = self.layers[-1]
            num_inputs = preceding_layer.num_nodes
        new_layer = Layer(num_nodes, num_inputs, activation_function)
        self.layers.append(new_layer)
        return new_layer
                
    def train(self, design_matrix, targets, eta=.01, eps=.01, max_iter=100):
        for iteration in range(max_iter):
            inputs = design_matrix
            for layer in self.layers:
                inputs = layer.feed_forward(inputs)

            cost_func_derivative = self._cost_func_derivatives[self.cost_function]
            cost_derivatives = cost_func_derivative(self, inputs, targets)        
            for layer in reversed(self.layers):
                deltas, weights = layer.back_propagate(cost_derivatives)
                cost_derivatives = deltas @ weights.T

            inputs = design_matrix
            for layer in self.layers:
                layer.update_weights(eta, inputs)
                layer.update_biases(eta)
                inputs = layer.outputs

            # print('outputs\n', layer.outputs)
            # print('weights\n',layer.weights)
            # print('biases\n',layer.biases)
        
        return layer.outputs
            
    def test(self, design_matrix):
        inputs = design_matrix
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return layer.outputs
        
            
        
