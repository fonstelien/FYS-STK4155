'''Implementation of the SGD class (Stochastic Gradient Descent)'''

import numpy as np
import sklearn as skl

class SGD:
    '''Class implementing Stochastic Gradient Descent for cost functions OLS, Ridge, ... on linear and logistic regression problems and classification problems.'''

    def _learning_schedule_constant(self, *args):
        return self.eta0

    def _learning_schedule_optimal(self, t, *args):
        return 1./(self.alpha*(self.t0 + t))

    def _learning_schedule_invscaling(self, t, *args):
        return self.eta0/(t+1)**.5

    _learning_schedules = {'constant':_learning_schedule_constant, 'optimal':_learning_schedule_optimal, 'invscaling':_learning_schedule_invscaling}

    def _grad_cost_function_ols(self, X, y, beta, *args):
        n, _ = X.shape
        return 2/n * X.T @ (X @ beta - y)

    def _grad_cost_function_ridge(self, X, y, beta, lmd, *args):
        n, _ = X.shape
        return 2/n * X.T @ (X @ beta - y) + 2*lmd*beta

    def _grad_cost_function_mse(self, X, y, beta, lmd, *args):
        n, _ = X.shape
        return 2/n * X.T @ (X @ beta - y) + 2*lmd*beta
    
    _grad_cost_functions = {'ols':_grad_cost_function_ols, 'ridge':_grad_cost_function_ridge, 'mse':_grad_cost_function_mse}
    
    def __init__(self, epochs, batches, eta0=.01, t0=None, alpha=1., learning_schedule='constant', cost_function='ols'):
        '''Initialize with number of epochs, number of mini batches, eta0, learning_schedule=['constant' | 'optimal' | 'invscaling'], t0 in 'optimal' schedule, cost_function=['ols' | 'ridge' | '']  '''
        self.epochs = epochs
        self.batches = batches
        self.eta0 = eta0
        self.t0 = t0 if t0 else epochs
        self.alpha = alpha
        self.learning_schedule = learning_schedule
        self.cost_function = cost_function
        self.beta = None

    def run(self, X, y, beta0=None, lmd=0.):
        '''Runs Stochastic Gradient Descent with inputs to cost function X, y, lmd. Starting point beta0 is set to all-zeros if None. lmd is Ridge's L2 hyperparameter and is used in learning_schedule 'optimal'. The coefficient vector is retrievable for later as self.beta. Returns the coefficient vector.'''
        n, p = X.shape
        n_batch = int(n/self.batches)
        self.beta = beta0 if beta0 else np.zeros((p,1))
        if (lmd != 0.):
            self.alpha = lmd
        
        schedule_func = self._learning_schedules[self.learning_schedule]
        grad_func = self._grad_cost_functions[self.cost_function]

        for epoch in range(self.epochs):
            X_shff, y_shff = skl.utils.resample(X, y, replace=False, random_state=epoch)
            for batch in range(self.batches):
                X_batch = X_shff[batch*n_batch:(batch+1)*n_batch]
                y_batch = y_shff[batch*n_batch:(batch+1)*n_batch]
                grad = grad_func(self, X_batch, y_batch, self.beta, lmd)
                eta = schedule_func(self, epoch*self.batches + batch)
                self.beta = self.beta - eta*grad

        return self.beta
