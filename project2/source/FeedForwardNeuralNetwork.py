'''Implementation of the FFNN class (Feed-Forward Neural Network)'''

import numpy as np
import sklearn as skl
from StochasticGradientDescent import SGD

class Layer:
    _leaky_relu_f = .01
    
    ## Activation functions and their derivatives
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s*(1-s)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return self._relu(z) / (z + (z == 0))  ## def. deriv. at 0 as 0

    def _leaky_relu(self, z):
        return np.maximum(self._leaky_relu_f*z, z)

    def _leaky_relu_derivative(self, z):
        return (z <= 0)*self._leaky_relu_f + (z > 0) ## def. deriv. at 0 as _leaky_relu_f

    
    _activation_funcs = {'sigmoid':_sigmoid,
                         'relu':_relu,
                         'leaky-relu':_leaky_relu}
    _activation_func_derivatives = {'sigmoid':_sigmoid_derivative,
                                    'relu':_relu_derivative,
                                    'leaky-relu':_leaky_relu_derivative}
    
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
    
    def update_weights(self, eta, lmd, inputs):
        self.weights = self.weights*(1-eta*lmd) - eta*(inputs.T @ self.deltas)
        return self.weights

    def update_biases(self, eta):
        self.biases = self.biases - eta*np.sum(self.deltas, axis=0)
        return self.biases
        
class FFNN:

    ## Learning schedules
    def _learning_schedule_constant(self, eta0, *args):
        return eta0

    def _learning_schedule_invscaling(self, eta0, t, *args):
        return eta0/(t+1)**.5

    def _learning_schedule_optimal(self, eta0, t, t0, alpha, *args):
        return 1./(alpha*(t0 + t))


    _learning_schedules = {'constant':_learning_schedule_constant,
                           'invscaling':_learning_schedule_invscaling,
                           'optimal':_learning_schedule_optimal}

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
    
    def train(self, features, targets, epochs=100, batches=100,
              learning_schedule='constant', eta0=.01, t0=None, lmd=None):
        n, p = features.shape
        n_batch = int(n/batches)

        cost_func_derivative = self._cost_func_derivatives[self.cost_function]
        schedule_func = self._learning_schedules[learning_schedule]
        
        if not t0:
            t0 = 1.
        alpha = eta0
        if lmd:
            alpha = lmd
        if not lmd:
            lmd = 0.
        
        for epoch in range(epochs):
            features_shff, targets_shff = skl.utils.resample(
                features, targets, replace=False, random_state=epoch)

            for batch in range(batches):
                features_batch = features_shff[batch*n_batch:(batch+1)*n_batch]
                targets_batch = targets_shff[batch*n_batch:(batch+1)*n_batch]
                
                ## Feed forward
                inputs = features_batch
                for layer in self.layers:
                    inputs = layer.feed_forward(inputs)

                ## Back-propagate
                cost_derivatives = cost_func_derivative(self, inputs, targets_batch)
                for layer in reversed(self.layers):
                    deltas, weights = layer.back_propagate(cost_derivatives)
                    cost_derivatives = deltas @ weights.T

                ## Update network
                t = epoch*batches + batch
                eta = schedule_func(self, eta0, t, t0, alpha)
                # print(eta)
                inputs = features_batch
                for layer in self.layers:
                    layer.update_weights(eta, lmd, inputs)
                    layer.update_biases(eta)
                    inputs = layer.outputs

                # print('outputs\n', layer.outputs)
                # print('weights\n',layer.weights)
                # print('biases\n',layer.biases)
        
        return layer.outputs
            
    def test(self, features):
        inputs = features
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return layer.outputs
        
            
        
