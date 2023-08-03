import numpy as np

def accuracy_FDA(y_true, y_pred):
    '''
    y_true and y_pred are both of type np.ndarray
    y_true (N, d) where N is the size of the validation set, and d is the dimension of the label
    y_pred (N, D) where N is the size of the validation set, and D is the output dimension of the ML model
    '''
    y_true = np.expand_dims(y_true, 1)
    if y_pred.shape[1] == 1:
        y_pred = np.round(y_pred)
    else:
        y_pred = y_pred.argmax(axis=1, keepdims=False)
    return 100*np.sum(y_pred==y_true)/y_pred.shape[0]