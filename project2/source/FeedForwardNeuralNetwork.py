'''Implementation of the FFNN class (Feed-Forward Neural Network) and Layer class which it uses for building the network.'''

import numpy as np
import sklearn as skl
from StochasticGradientDescent import SGD

class Layer:
    '''Class implementing layer of nodes in a feed-forward neural network. Works well with the FFNN class.'''
    _leaky_relu_f = .01
    
    ## Activation functions and their derivatives
    def _sigmoid(self, z):
        stability_factor = 1E2
        z_stable = np.minimum(z, stability_factor)
        e = np.exp(z_stable)
        return e/(e + 1)
    
    def _sigmoid_derivative(self, z):
        a = self._sigmoid(z)
        return a*(1 - a)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0)*1  ## def. deriv. at 0 as 0

    def _leaky_relu(self, z):
        return np.maximum(self._leaky_relu_f*z, z)

    def _leaky_relu_derivative(self, z):
        return (z <= 0)*self._leaky_relu_f + (z > 0) ## def. deriv. at 0 as _leaky_relu_f

    def _softmax(self, z):
        stability_factor = z.max()
        e = np.exp(z-stability_factor)
        return e/np.sum(e, axis=1, keepdims=True)

    def _softmax_derivative(self, z):
        a = self._softmax(z)
        return a*(1 - a)

    def _unit_step(self, z):
        return (z > 0)*1

    def _unit_step_derivative(self, z):
        return np.ones(z.shape)
    
    _activation_funcs = {'sigmoid':_sigmoid,
                         'relu':_relu,
                         'leaky-relu':_leaky_relu,
                         'softmax':_softmax,
                         'unit-step':_unit_step}
    _activation_func_derivatives = {'sigmoid':_sigmoid_derivative,
                                    'relu':_relu_derivative,
                                    'leaky-relu':_leaky_relu_derivative,
                                    'softmax':_softmax_derivative,
                                    'unit-step':_unit_step_derivative}
    
    def __init__(self, num_nodes, num_inputs, activation_function='sigmoid', init_biases=None, init_weights=None):
        '''Initialize with number of nodes in the layer and number of inputs to each node from preceding layer; activation_function=['sigmoid' | 'relu' | 'leaky-relu' | 'softmax' | 'unit-step']. Some default initial biases and weights are used if init_biases and init_weights are None.'''
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.init_biases = init_biases
        self.init_weights = init_weights
        self.biases = None
        self.weights = None
        self.reset()  # initialize biases, weights
        self.activation_inputs = None  # matrix (num_samples x num_nodes)
        self.outputs = None  # matrix (num_samples x num_nodes)
        self.deltas = None  # matrix (num_samples x num_nodes)

    def reset(self):
        '''Resets layer to initial untrained state.'''
        np.random.seed(0)
        if self.init_biases:
            self.biases = self.init_biases
        else:
            self.biases = np.zeros(self.num_nodes) + .01

        if self.init_weights:
            self.weights = self.init_weights
        else:
            self.weights = np.random.rand(self.num_inputs, self.num_nodes)
    
    def feed_forward(self, inputs):
        '''Feed-forward step for layer on the inputs given as argument. Returns the output of this layer.'''
        activation_func = self._activation_funcs[self.activation_function]
        self.activation_inputs = inputs @ self.weights + self.biases
        self.outputs = activation_func(self, self.activation_inputs)
        return self.outputs

    def back_propagate(self, error):
        '''Back-propagate the error through this layer to calculate the deltas. Returns (deltas, weights).'''
        activation_func_derivative = self._activation_func_derivatives[self.activation_function]
        activation_derivatives = activation_func_derivative(self, self.activation_inputs)
        self.deltas = error * activation_derivatives
        return (self.deltas, self.weights)
    
    def update_weights(self, eta, lmd, inputs):
        '''Update weights of this layer using gradient descent with step size eta and L2 regularization lmd'''
        self.weights = self.weights*(1-eta*lmd) - eta*(inputs.T @ self.deltas)
        return self.weights

    def update_biases(self, eta):
        '''Update biases of this layer using gradient descent with step size eta.'''
        self.biases = self.biases - eta*np.sum(self.deltas, axis=0)
        return self.biases
        
class FFNN:
    '''Class implementing a feed-forward neural network. Works well with the Layer class.'''

    ## Learning schedules
    def _learning_schedule_constant(self, eta0, *args):
        return eta0

    def _learning_schedule_invscaling(self, eta0, t, *args):
        return eta0/(t+1)**.5

    def _learning_schedule_geron(self, eta0, t, t0, *args):
        return eta0/(t0 + t)

    _learning_schedules = {'constant':_learning_schedule_constant,
                           'invscaling':_learning_schedule_invscaling,
                           'geron':_learning_schedule_geron}

    ## Cost function derivatives
    def _mse_derivative(self, a, t):
        return a - t

    def _cross_entropy_derivative(self, a, t):
        stability_factor = 1E-9
        return (a - t)/((a + stability_factor)*(1 - a + stability_factor))

    

    _cost_func_derivatives = {'mse':_mse_derivative,
                              'cross-entropy':_cross_entropy_derivative}
    
    def __init__(self, num_features, cost_function='mse'):
        '''Initialize with number of features num_features and cost_function=['mse' | 'cross-entropy']'''
        self.num_features = num_features
        self.cost_function = cost_function
        self.layers = list()

    def add_layer(self, num_nodes, activation_function='sigmoid', init_biases=None, init_weights=None):
        '''Instantiates and appends to the FFNN a layer from the Layer class with num_nodes nodes and activation_function=['sigmoid' | 'relu' | 'leaky-relu' | 'softmax' | 'unit-step']. Some default biases and weights defined in Layer class are used if init_biases and init_weights are None.'''
        num_inputs = self.num_features
        if (len(self.layers) > 0):
            preceding_layer = self.layers[-1]
            num_inputs = preceding_layer.num_nodes
        new_layer = Layer(num_nodes, num_inputs, activation_function=activation_function,
                          init_biases=init_biases, init_weights=init_weights)
        self.layers.append(new_layer)
        return new_layer
    
    def train(self, data, targets, epochs=100, batch_size=100, batches=None,
              learning_schedule='constant', eta0=.01, t0=1., lmd=.0):
        '''Trains the FFNN on the data and targets given as argument. All data structures are np.ndarray type. Trains using stochastic gradient descent over epochs number of epochs in batches defined by batches or batch_size; learning_schedule=['constant' | 'invscaling' | 'geron']; step size eta0; t0 used in some learning schedules; L2 regularization with lmd. Returns the output layer's output.'''
        n, p = data.shape
        if batches:
            batch_size = int(n/batches)
        else:
            batches = int(n/batch_size)
        
        cost_func_derivative = self._cost_func_derivatives[self.cost_function]
        schedule_func = self._learning_schedules[learning_schedule]

        for epoch in range(epochs):
            data_shff, targets_shff = skl.utils.resample(
                data, targets, replace=False, random_state=epoch)

            for batch in range(batches):
                data_batch = data_shff[batch*batch_size:(batch+1)*batch_size]
                targets_batch = targets_shff[batch*batch_size:(batch+1)*batch_size]

                ## Feed forward
                inputs = data_batch
                for layer in self.layers:
                    inputs = layer.feed_forward(inputs)

                ## Back-propagate
                error = cost_func_derivative(self, inputs, targets_batch)
                for layer in reversed(self.layers):
                    deltas, weights = layer.back_propagate(error)
                    error = deltas @ weights.T

                ## Update network
                t = epoch*batches + batch
                eta = schedule_func(self, eta0, t, t0, lmd)
                inputs = data_batch
                for layer in self.layers:
                    layer.update_weights(eta, lmd, inputs)
                    layer.update_biases(eta)
                    inputs = layer.outputs

        return layer.outputs

    def predict(self, data):
        '''Makes a prediction on the data given as argument and returns the prediction. All data structures are np.ndarray type.'''
        inputs = data
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return layer.outputs

    def reset(self):
        '''Resets the FFNN's layers to initial untrained states.'''
        for layer in self.layers:
            layer.reset()
