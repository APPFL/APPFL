import numpy as np


def accuracy(y_true, y_pred):
    if len(y_pred.shape) == 1:
        y_pred = np.round(y_pred)
    else:
        y_pred = y_pred.argmax(axis=1)
    return 100 * np.sum(y_pred == y_true) / y_true.shape[0]
