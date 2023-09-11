from .cnn import CNN
from .resnet import resnet18
from .lstm import LSTMForecast

def get_model(args):
    ## User-defined model
    model = None
    if args.model == "CNN":
        model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "resnet18":
        model = resnet18(num_channel=args.num_channel, num_classes=args.num_classes, pretrained=args.pretrained)  
    if args.model == "LSTM":
        model = LSTMForecast(n_features=args.n_features,n_lookback=args.n_lookback,n_lstm_layers=args.n_lstm_layers,n_hidden_size=args.n_hidden_size)
    return model

def validate_parameter_names(model,list_of_params):
    
    # inputs model and list with name of parameters
    # returns IS_VALID, IS_EMPTY

    if list_of_params == []:
        IS_VALID = True
        IS_EMPTY = True
        return IS_VALID, IS_EMPTY
    
    model_keys = [key for key,_ in model.named_parameters()]
    
    for p in list_of_params:
        if not(p in model_keys):
            IS_VALID = False
            IS_EMPTY = False
            return IS_VALID,IS_EMPTY
    
    IS_VALID = True
    IS_EMPTY = False
    return IS_VALID,IS_EMPTY