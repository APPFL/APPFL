import numpy as np

def accuracy(y_true, y_pred):
    '''
    y_true and y_pred are both of type np.ndarray
    y_true (N, d) where N is the size of the validation set, and d is the dimension of the label
    y_pred (N, D) where N is the size of the validation set, and D is the output dimension of the ML model
    '''
    if len(y_pred.shape) == 1:
        y_pred = np.round(y_pred)
    else:
        y_pred = y_pred.argmax(axis=1)
    return 100*np.sum(y_pred==y_true)/y_pred.shape[0]

def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).mean()
    except ValueError:
        return np.nan
