import math
import torch
import einops

import numpy as np


##############
# ShuffleNet #
##############
class ShuffleNetInvRes(torch.nn.Module):
    def __init__(self, inp, oup, stride, branch):
        super(ShuffleNetInvRes, self).__init__()
        self.branch = branch
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2
        if self.branch == 1:
        	self.branch2 = torch.nn.Sequential(
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
            )                
        else:                  
            self.branch1 = torch.nn.Sequential(
                torch.nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                torch.nn.BatchNorm2d(inp),
                torch.nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
            )        
            self.branch2 = torch.nn.Sequential(
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
            )        

    def forward(self, x):
        if self.branch == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        elif self.branch == 2:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        B, C, H, W = out.size()
        channels_per_group = C // 2
        out = out.view(B, 2, channels_per_group, H, W)
        out = torch.transpose(out, 1, 2).contiguous()
        out = out.view(B, -1, H, W)
        return out

##########################
# MobileNet & MobileNeXt #
##########################
def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(torch.nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, make_divisible(in_channels // reduction, 8)),
            torch.nn.ReLU(True),
            torch.nn.Linear(make_divisible(in_channels // reduction, 8), in_channels),
            torch.nn.Hardsigmoid(True)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, inputs, hidden_dims, ouputs, kernel_size, stride, use_se, use_hardswish):
        super(InvertedResidualBlock, self).__init__()
        self.identity = stride == 1 and inputs == ouputs

        if inputs == hidden_dims:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dims, bias=False), # depth-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                SELayer(hidden_dims) if use_se else torch.nn.Identity(), # squeeze-excite block
                torch.nn.Conv2d(hidden_dims, ouputs, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(ouputs),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inputs, hidden_dims, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                torch.nn.Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dims, bias=False), # depth-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                SELayer(hidden_dims) if use_se else torch.nn.Identity(), # squeeze-excite block
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                torch.nn.Conv2d(hidden_dims, ouputs, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(ouputs),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SandGlassLayer(torch.nn.Module):
    def __init__(self, inputs, outputs, stride, reduction_ratio):
        super(SandGlassLayer, self).__init__()
        hidden_dim = round(inputs // reduction_ratio)
        self.identity = stride == 1 and inputs == outputs

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(inputs, inputs, 3, 1, 1, groups=inputs, bias=False), # depth-wise convolution
            torch.nn.BatchNorm2d(inputs),
            torch.nn.ReLU6(True),
            torch.nn.Conv2d(inputs, hidden_dim, 1, 1, 0, bias=False), # point-wise convolution
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Conv2d(hidden_dim, outputs, 1, 1, 0, bias=False), # point-wise convolution
            torch.nn.BatchNorm2d(outputs),
            torch.nn.ReLU6(True),
            torch.nn.Conv2d(outputs, outputs, 3, stride, 1, groups=outputs, bias=False), # depth-wise convolution
            torch.nn.BatchNorm2d(outputs),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
##############
# SqueezeNet #
##############
class FireBlock(torch.nn.Module):
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireBlock, self).__init__()
        self.squeeze_activation = torch.nn.ReLU(True)
        self.in_planes = in_planes
        self.squeeze = torch.nn.Conv2d(in_planes, squeeze_planes, kernel_size=1)
        self.expand1x1 = torch.nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = torch.nn.ReLU(True)
        self.expand3x3 = torch.nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)

###############
# SqueezeNeXt #
###############
class SNXBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction=0.5):
        super(SNXBlock, self).__init__()
        if stride == 2:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
        
        self.act = torch.nn.ReLU()
        self.squeeze = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction * 0.5)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

        if stride == 2 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Identity()
            
    def forward(self, x):
        out = self.squeeze(x)
        out = out + self.act(self.shortcut(x))
        out = self.act(out)
        return out

#############
# MobileViT #
#############
class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.SiLU(True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ff(x)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**(-0.5)

        self.attend = torch.nn.Softmax(dim=-1)
        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(torch.nn.Module):
    def __init__(self, inputs, ouputs, stride=1, expansion=4):
        super(MV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inputs * expansion)
        self.use_res_connect = self.stride == 1 and inputs == ouputs

        if expansion == 1:
            self.conv = torch.nn.Sequential(
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, ouputs, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(ouputs),
            )
        else:
            self.conv = torch.nn.Sequential(
                # pw
                torch.nn.Conv2d(inputs, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(True),
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, ouputs, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(ouputs),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTBlock(torch.nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.patch_size = patch_size

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel, kernel_size, 1, 1, bias=False),
            torch.nn.BatchNorm2d(channel),
            torch.nn.SiLU(True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(channel, dim, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(dim),
            torch.nn.SiLU(True)
        )
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(dim, channel, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(channel),
            torch.nn.SiLU(True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(2 * channel, channel, kernel_size, 1, 1, bias=False),
            torch.nn.BatchNorm2d(channel),
            torch.nn.SiLU(True)
        )
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = einops.rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.patch_size, pw=self.patch_size)
        x = self.transformer(x)
        x = einops.rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.patch_size, w=w // self.patch_size, ph=self.patch_size, pw=self.patch_size)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

##########
# ResNet #
##########
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.GroupNorm(planes // 2, planes),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.GroupNorm(planes // 2, planes)
        )

        self.shortcut = torch.nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.GroupNorm(planes // 2, planes)
            )

    def forward(self, x):
        x = self.features(x) + self.shortcut(x) 
        x = torch.nn.functional.relu(x)
        return x

################
# Lambda Layer #
################
class Lambda(torch.nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x): 
        return self.func(x)

#############################
# Positional Encoding Layer #
#############################
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len=10000):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x

###################
# UNet3DConvBlock #
###################
class UNet3DConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalization=None, kernel_size=3, activation='ReLU', preactivation=False, padding=0, padding_mode='zeros', dilation=None, dropout=0):
        super(UNet3DConvBlock, self).__init__()
        block = torch.nn.ModuleList()

        # dilation layer
        dilation = 1 if dilation is None else dilation

        # padding
        if padding:
            total_padding = kernel_size + 2 * (dilation - 1) - 1
            padding = total_padding // 2

        # bias
        no_bias = not preactivation and (normalization is not None)

        # convolution
        conv_layer = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=not no_bias,
        )

        # normalization
        norm_layer = None
        if normalization is not None:
            class_name = f'{normalization.capitalize()}Norm3d'
            norm_class = getattr(torch.nn, class_name)
            num_features = in_channels if preactivation else out_channels
            norm_layer = norm_class(num_features)

        # activation
        activation_layer = None
        if activation is not None:
            activation_layer = getattr(torch.nn, activation)()

        # preactivation
        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, conv_layer)
        else:
            self.add_if_not_none(block, conv_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        # dropout
        dropout_layer = None
        if dropout:
            dropout_class = torch.nn.Dropout3d
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        # aggregate together
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer
        self.block = torch.nn.Sequential(*block)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)

    def forward(self, x):
        return self.block(x)

#################
# UNet3DEncoder #
#################
class UNet3DEncodingBlock(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels_first, normalization, pooling_type,
        preactivation=False, is_first_block=False, residual=False, padding=0, padding_mode='zeros', activation='ReLU', dilation=None, dropout=0
        ):
        super(UNet3DEncodingBlock, self).__init__()
        self.preactivation = preactivation
        self.normalization = normalization
        self.residual = residual

        if is_first_block:
            normalization = None
            preactivation = None
        else:
            normalization = self.normalization
            preactivation = self.preactivation

        self.conv1 = UNet3DConvBlock(
            in_channels,
            out_channels_first,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        out_channels_second = 2 * out_channels_first
        self.conv2 = UNet3DConvBlock(
            out_channels_first,
            out_channels_second,
            normalization=self.normalization,
            preactivation=self.preactivation,
            padding=padding,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        if residual:
            self.conv_residual = UNet3DConvBlock(
                in_channels,
                out_channels_second,
                kernel_size=1,
                normalization=None,
                activation=None,
            )
        self.downsample = None
        if pooling_type is not None:
            class_name = f'{pooling_type.capitalize()}Pool3d'
            pool_layer = getattr(torch.nn, class_name)
            self.downsample = pool_layer(kernel_size=2)

    @property
    def out_channels(self):
        return self.conv2.conv_layer.out_channels

    def forward(self, x):
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
            return x, skip_connection

class UNet3DEncoder(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels_first, pooling_type, num_encoding_blocks, normalization,
        preactivation=False, residual=False, padding=0, padding_mode='zeros', activation='ReLU', initial_dilation=None, dropout=0
    ):
        super(UNet3DEncoder, self).__init__()
        self.encoded_blocks = torch.nn.ModuleList()

        self.dilation = initial_dilation
        is_first_block = True
        for _ in range(num_encoding_blocks):
            encoded_block = UNet3DEncodingBlock(
                in_channels,
                out_channels_first,
                normalization,
                pooling_type,
                preactivation,
                is_first_block=is_first_block,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            is_first_block = False
            self.encoded_blocks.append(encoded_block)

            in_channels = 2 * out_channels_first
            out_channels_first = in_channels
            if self.dilation is not None:
                self.dilation *= 2
    
    @property
    def out_channels(self):
        return self.encoded_blocks[-1].out_channels

    def forward(self, x):
        skip_connections = []
        for encoded_block in self.encoded_blocks:
            x, skip_connnection = encoded_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

#################
# UNet3DDecoder #
#################
class UNet3DDecodingBlock(torch.nn.Module):
    def __init__(
        self,in_channels, upsampling_type, normalization,
        preactivation=True, residual=False, padding=0, padding_mode='zeros', activation='ReLU', dilation=None, dropout=0
    ):
        super(UNet3DDecodingBlock, self).__init__()
        self.residual = residual

        if upsampling_type == 'conv':
            in_channels = out_channels = 2 * in_channels
            self.upsample = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.upsample = torch.nn.Upsample(scale_factor=2, mode=upsampling_type, align_corners=False)

        in_channels_first = in_channels * (1 + 2)
        out_channels = in_channels
        self.conv1 = UNet3DConvBlock(
            in_channels_first,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        in_channels_second = out_channels
        self.conv2 = UNet3DConvBlock(
            in_channels_second,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = UNet3DConvBlock(
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2

        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = torch.nn.functional.pad(skip_connection, pad.tolist())
        return skip_connection

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=1)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x

class UNet3DDecoder(torch.nn.Module):
    def __init__(
        self, in_channels, upsampling_type, num_decoding_blocks, normalization,
        preactivation=False, residual=False, padding=0, padding_mode='zeros', activation='ReLU', initial_dilation=None, dropout=0
    ):
        super(UNet3DDecoder, self).__init__()
        upsampling_type = 'trilinear' if upsampling_type == 'linear' else upsampling_type
        
        self.decoding_blocks = torch.nn.ModuleList()
        self.dilation = initial_dilation
        for _ in range(num_decoding_blocks):
            decoding_block = UNet3DDecodingBlock(
                in_channels,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels //= 2
            if self.dilation is not None:
                self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x

#########
# DANet #
#########
# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(torch.autograd.Function):
    # https://github.com/WhatAShot/DANet/blob/main/model/sparsemax.py
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size

class Sparsemax(torch.nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()
        self.sparsemax = SparsemaxFunction.apply
        
    def forward(self, input):
        return self.sparsemax(input, self.dim)

class Entmax15Function(torch.autograd.Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size

class Entmoid15(torch.autograd.Function):
    """ A highly optimized equivalent of lambda x: Entmax15([x, 0]) """

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(torch.nn.functional.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * torch.nn.functional.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input

entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply


class Entmax15(torch.nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)

class LearnableLocality(torch.nn.Module):
    def __init__(self, input_dim, k):
        super(LearnableLocality, self).__init__()
        self.register_parameter('weight', torch.nn.Parameter(torch.rand(k, input_dim)))
        self.smax = Entmax15(dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)  # [B, k, D]
        return masked_x

class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """
    def __init__(self, input_dim, virtual_batch_size=512):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = torch.nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training == True:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

class AbstractLayer(torch.nn.Module):
    def __init__(self, base_input_dim, base_output_dim, k, virtual_batch_size, bias=True):
        super(AbstractLayer, self).__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, k=k)
        self.fc = torch.nn.Conv1d(base_input_dim * k, 2 * k * base_output_dim, kernel_size=1, groups=k, bias=bias)
        self.initialize_glu(self.fc, input_dim=base_input_dim * k, output_dim=2 * k * base_output_dim)
        self.bn = GBN(2 * base_output_dim * k, virtual_batch_size)
        self.k = k
        self.base_output_dim = base_output_dim

    def initialize_glu(self, module, input_dim, output_dim):
        gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
        torch.nn.init.xavier_normal_(module.weight, gain=gain_value)

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, k, D]
        x = self.fc(x.view(b, -1, 1))  # [B, k, D] -> [B, k * D, 1] -> [B, k * (2 * D'), 1]
        x = self.bn(x)
        chunks = x.chunk(self.k, 1)  # k * [B, 2 * D', 1]
        x = sum([torch.nn.functional.relu(torch.sigmoid(x_[:, :self.base_output_dim, :]) * x_[:, self.base_output_dim:, :]) for x_ in chunks])  # k * [B, D', 1] -> [B, D', 1]
        return x.squeeze(-1)

class DANetBlock(torch.nn.Module):
    def __init__(self, input_dim, base_outdim, k, virtual_batch_size, fix_input_dim, drop_rate):
        super(DANetBlock, self).__init__()
        self.conv1 = AbstractLayer(input_dim, base_outdim // 2, k, virtual_batch_size)
        self.conv2 = AbstractLayer(base_outdim // 2, base_outdim, k, virtual_batch_size)

        self.downsample = torch.nn.Sequential(
            torch.nn.Dropout(drop_rate),
            AbstractLayer(fix_input_dim, base_outdim, k, virtual_batch_size)
        )

    def forward(self, x, pre_out=None):
        if pre_out == None:
            pre_out = x
        out = self.conv1(pre_out)
        out = self.conv2(out)
        identity = self.downsample(x)
        out += identity
        return torch.nn.functional.leaky_relu(out, 0.01)
