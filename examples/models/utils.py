from .cnn import CNN, ComplexCNN
from .resnet import resnet18


def get_model(args):
    ## User-defined model
    if args.model == "CNN":
        model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "ComplexCNN":
        model = ComplexCNN(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "resnet18":
        model = resnet18(
            num_channel=args.num_channel,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
        )
    return model
