'''Neural network running methods'''

from utils import *


def run_neural_network_kfold(X, t, FFNN, t_onehot=None, k=5, etas=[], lambdas=[], **kwargs):
    '''Trains the FFNN with k-fold resampling on X, t and looping over etas, lambdas. Use one-hot representation of t in t_onehot for classification problems. kwargs are forwarded to FFNN.train(). Returns 2D np.ndarrays (train_accuracy, test_accuracy) indexed on eta,lmd.'''

    splits = split(t, k)
    train_accuracy = np.ndarray((len(etas), len(lambdas)))
    test_accuracy = np.ndarray((len(etas), len(lambdas)))
    train_accuracy_buf = np.ndarray(k)
    test_accuracy_buf = np.ndarray(k)

    for i, eta in enumerate(etas):
        for j, lmd in enumerate(lambdas):
            for s, (train_split, test_split) in enumerate(zip(*splits)):
                X_train = X[train_split]
                t_train = t[train_split]
                train_targets = t_train if t_onehot is None else t_onehot[train_split]

                X_test = X[test_split]
                t_test = t[test_split]

                X_train, X_test = scale(X_train, X_test, with_std=False, scale_intercept=True)

                FFNN.train(X_train, train_targets, eta0=eta, lmd=lmd, **kwargs)
                pred = FFNN.predict(X_train)
                if not t_onehot is None:
                    pred = pred.argmax(axis=1)
                train_accuracy_buf[s] = skl.metrics.accuracy_score(t_train, pred)

                pred = FFNN.predict(X_test)
                if not t_onehot is None:
                    pred = pred.argmax(axis=1)
                test_accuracy_buf[s] = skl.metrics.accuracy_score(t_test, pred)


                FFNN.reset()

            train_accuracy[i,j] = np.mean(train_accuracy_buf)
            test_accuracy[i,j] = np.mean(test_accuracy_buf)

    return (train_accuracy, test_accuracy)
