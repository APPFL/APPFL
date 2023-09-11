import numpy as np

def accuracy(y_true, y_pred):
    '''
    y_true and y_pred are both of type np.ndarray
    y_true (N, d) where N is the size of the validation set, and d is the dimension of the label
    y_pred (N, D) where N is the size of the validation set, and D is the output dimension of the ML model
    '''
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()
    
    numerator = np.mean( np.abs(y_true-y_pred) )
    denominator = np.mean( np.abs(np.diff(y_pred,n=1)) )
    return np.divide(numerator,denominator)
