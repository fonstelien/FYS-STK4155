'''Implementation of the SGD class (Stochastic Gradient Descent)'''

import numpy as np
import sklearn as skl

class SGD:
    '''Class implementing Stochastic Gradient Descent for cost functions OLS, Ridge, ... on linear and logistic regression problems and classification problems.'''

    ## Learning schedules    
    def _learning_schedule_constant(self, *args):
        return self.eta0

    def _learning_schedule_invscaling(self, t, *args):
        return self.eta0/(t+1)**.5

    def _learning_schedule_geron(self, t, *args):
        return self.eta0/(self.t0 + t)
    
    _learning_schedules = {'constant':_learning_schedule_constant,
                           'geron':_learning_schedule_geron,
                           'invscaling':_learning_schedule_invscaling}

    ## Cost function gradients    
    def _mse_grad(self, X, y, beta, lmd, *args):
        n, _ = X.shape
        return 2/n * X.T @ (X @ beta - y) + 2*lmd*beta

    def _cross_entropy_grad(self, X, y, beta, lmd, *args):
        e = np.exp(X @ beta)
        mu = e/np.sum(e, axis=1, keepdims=True)
        n, _ = X.shape        
        return 1/n * X.T @ (mu - y) + lmd*beta
    
    _grad_cost_functions = {'linear':_mse_grad,
                            'logistic':_cross_entropy_grad}

    def __init__(self, epochs=100, batch_size=100, batches=None, eta0=.01, t0=1, learning_schedule='constant', regression='linear'):
        '''Initialize with number of epochs, number of mini batches, eta0, t0 in 'geron' schedule, learning_schedule=['constant' | 'geron' | 'invscaling'], schedule, regression=['linear' | 'logistic'].'''
        self.epochs = epochs
        self.batch_size = batch_size        
        self.batches = batches
        self.eta0 = eta0
        self.t0 = t0
        self.learning_schedule = learning_schedule
        self.regression = regression
        self.beta = None

    def run(self, X, y, beta0=None, eta0=None, lmd=0.):
        '''Runs Stochastic Gradient Descent with inputs to cost function X, y, lmd. Starting point beta0 is set to all-zeros if None. L2 regularization factor lmd. The coefficient vector is retrievable for later as self.beta. Returns the coefficient vector as np.ndarray.'''
        n, p = X.shape

        if self.batches:
            self.batch_size = int(n/self.batches)
        else:
            self.batches = int(n/self.batch_size)

        self.beta = beta0
        if beta0 is None:
            if self.regression == 'logistic':
                _, classes = y.shape
                self.beta = np.zeros((p,classes))                
            else:
                self.beta = np.zeros((p,1))

        if eta0:
            self.eta0 = eta0
                
        schedule_func = self._learning_schedules[self.learning_schedule]
        grad_func = self._grad_cost_functions[self.regression]

        for epoch in range(self.epochs):
            X_shff, y_shff = skl.utils.resample(X, y, replace=False, random_state=epoch)
            for batch in range(self.batches):
                X_batch = X_shff[batch*self.batch_size:(batch+1)*self.batch_size]
                y_batch = y_shff[batch*self.batch_size:(batch+1)*self.batch_size]
                grad = grad_func(self, X_batch, y_batch, self.beta, lmd)
                eta = schedule_func(self, epoch*self.batches + batch)
                self.beta = self.beta - eta*grad

        return self.beta
