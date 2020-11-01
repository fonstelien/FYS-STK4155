'''Implementation of the FFNN class (Feed-Forward Neural Network)'''

import numpy as np
import sklearn as skl
from StochasticGradientDescent import SGD

class Layer:

    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def _derivative_sigmoid(self, z):
        s = self._sigmoid(z)
        return s*(1-s)

    
    _activation_funcs = {'sigmoid':_sigmoid}
    _activation_func_derivatives = {'sigmoid':_derivative_sigmoid}
    
    def __init__(self, num_nodes, num_inputs, activation_function='sigmoid'):
        self.biases = np.ones((num_nodes, 1))
        self.weights = np.random.rand(num_inputs, num_nodes)
        self.output_vals = np.ndarray((num_nodes, 1))
        self.z = np.ndarray((num_nodes, 1))
        self.deltas = np.ndarray((num_nodes, 1))
        self.activation_function = activation_function
        
    def feed_forward(self, input_vals):
        activation_func = self._activation_funcs[self.activation_function]
        self.z[:,0] = (self.weights.T @ input_vals + self.biases).ravel()
        self.output_vals[:,0] = (activation_func(self, self.z)).ravel()
        return self.output_vals

    def back_propagate(self, weights, deltas):
        activation_func_derivative = self._activation_func_derivatives[self.activation_function]
        self.derivatives = activation_func_derivative(self, self.z)
        self.deltas[:,0] = (self.derivatives.T @ weights @ deltas).ravel()
        return (self.weights, self.deltas)

    def update_weights(self):
        pass

    def update_biases(self):
        pass
    
class FFNN:

    def __init__(self, num_features, cost_function='mse'):
        self.num_features = num_features
        self.layers = list()
        
