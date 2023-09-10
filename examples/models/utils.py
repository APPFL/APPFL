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
        model = LSTMForecast(n_features=args.n_features,n_lookback=args.lookback,n_lstm_layers=args.lstm_layers,n_hidden_size=args.lstm_hidden_size)
    return model
