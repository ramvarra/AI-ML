'''
Module to Save and perform inference using a model trained in Keras, but using only numpy functions.
Assumes  hidden Dense layers using relu activations
Assumes  binary classification output (sigmoid)
'''

import numpy as np
import logging

def save_weights(model, file_name):
    '''
    Save weights in a keras model for Dense layers.
    Asssumes the model trainable parameters are Dense layers (ignores Dropouts)
    '''
    weights = []
    for layer in model.layers:
        cfg = layer.get_config()
        #print(f'Layer {l}: ', cfg)
        if 'units' in cfg:
            w, b = layer.get_weights()
            weights.append([w, b])
    logging.debug(f'Saving weights to: {file_name}')
    np.save(file_name, weights)

def load_weights(file_name):
    logging.debug(f'Loading weights from: {file_name}')
    return np.load(file_name)


def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))

def predict_np(X, weights):
    assert len(X.shape) == 1
    Z = X
    for (i, (W, b)) in enumerate(weights):
        assert len(W.shape) == 2 and len(b.shape) == 1
        assert Z.shape[0] == W.shape[0]
        assert W.shape[1] == b.shape[0]
        Z = np.dot(Z, W) + b
        if i == len(weights)-1:  # if last layer, apply sigmoid
            assert Z.shape == (1,)
            Z = sigmoid_np(Z)[0]
        else:
            Z[Z<0] = 0  # relu
    return Z
