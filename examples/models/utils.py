from .cnn import CNN
from .resnet import ResNet18
from .lstm import LSTMForecast
try:
    import flamby.datasets.fed_ixi as IXI
    import flamby.datasets.fed_kits19 as Kits19
    import flamby.datasets.fed_tcga_brca as TcgaBrca
    import flamby.datasets.fed_isic2019 as ISIC2019
    import flamby.datasets.fed_heart_disease as HeartDisease
except:
    print("FLamby is not installed.")

def get_model(args):
    ## User-defined model
    model = None
    if args.model == "CNN":
        model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    elif args.model == "resnet18":
        model = ResNet18()
    elif args.model == "LSTM":
        model = LSTMForecast(n_features=args.n_features,n_lookback=args.n_lookback,n_lstm_layers=args.n_lstm_layers,n_hidden_size=args.n_hidden_size)
    else:
        raise NotImplementedError
    return model

def flamby_train(dataset: str):
    if dataset == 'TcgaBrca':
        return TcgaBrca.Baseline(), TcgaBrca.BaselineLoss(), \
            'Adam', TcgaBrca.LR, TcgaBrca.BATCH_SIZE, TcgaBrca.metric
    elif dataset == 'HeartDisease':
        return HeartDisease.Baseline(), HeartDisease.BaselineLoss(), \
            'Adam', HeartDisease.LR, HeartDisease.BATCH_SIZE, HeartDisease.metric
    elif dataset == 'IXI':
        return IXI.Baseline(), IXI.BaselineLoss(), \
            'AdamW', IXI.LR, IXI.BATCH_SIZE, IXI.metric
    elif dataset == 'ISIC2019':
        return ISIC2019.Baseline(), ISIC2019.BaselineLoss(), \
            'Adam', ISIC2019.LR, ISIC2019.BATCH_SIZE, ISIC2019.metric
    elif dataset == 'Kits19':
        return Kits19.Baseline(), Kits19.BaselineLoss(), \
            'Adam', Kits19.LR, Kits19.BATCH_SIZE, Kits19.metric
    else:
        raise NotImplementedError

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