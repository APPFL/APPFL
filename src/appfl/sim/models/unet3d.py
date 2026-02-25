import torch

from .model_utils import UNet3DEncoder, UNet3DEncodingBlock, UNet3DDecoder, UNet3DConvBlock



class UNet3D(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, num_classes, num_layers, dropout):
        super(UNet3D, self).__init__()
        # UNet3Dencoder
        self.UNet3Dencoder = UNet3DEncoder(
            in_channels=in_channels,
            out_channels_first=hidden_size,
            pooling_type='max',
            num_encoding_blocks=num_layers - 1,
            normalization='batch',
            preactivation=False,
            residual=False,
            padding=1,
            padding_mode='zeros',
            activation='PReLU',
            initial_dilation=None,
            dropout=dropout,
        )

        # bottom (last encoded block)
        in_channels = self.UNet3Dencoder.out_channels
        out_channels_first = in_channels

        self.bottom_block = UNet3DEncodingBlock(
            in_channels=in_channels,
            out_channels_first=out_channels_first,
            normalization='batch',
            pooling_type=None,
            preactivation=False,
            residual=False,
            padding=1,
            padding_mode='zeros',
            activation='ReLU',
            dilation=None,
            dropout=dropout,
        )

        # UNet3Ddecoder
        in_channels = self.bottom_block.out_channels
        in_channels = hidden_size * 2**(num_layers - 1)
        num_decoding_blocks = num_layers - 1
        self.UNet3Ddecoder = UNet3DDecoder(
            in_channels=in_channels,
            upsampling_type='linear',
            num_decoding_blocks=num_decoding_blocks,
            normalization='batch',
            preactivation=False,
            residual=False,
            padding=1,
            padding_mode='zeros',
            activation='ReLU',
            initial_dilation=None,
            dropout=dropout,
        )

        # classifier
        self.classifier = UNet3DConvBlock(
            in_channels=2 * hidden_size, 
            out_channels=num_classes,
            kernel_size=1, 
            activation=None
        )

    def forward(self, x):
        skip_connections, encoded = self.UNet3Dencoder(x)
        encoded = self.bottom_block(encoded)
        x = self.UNet3Ddecoder(skip_connections, encoded)
        x = self.classifier(x).softmax(dim=1)
        return x
