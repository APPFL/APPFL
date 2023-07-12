from appfl.compressor.compressor import *
import numpy as np
from appfl.config import *
import torch
import torch.nn as nn
import math
import appfl.misc.utils as utils
import copy


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        ###
        ### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
        ###
        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def test_basic_compress(cfg: Config) -> None:
    # Create a compressor
    compressor = Compressor(cfg)

    # Create a random 1D array
    ori_data = np.random.rand(1000)
    ori_shape = ori_data.shape
    ori_dtype = ori_data.dtype

    # Compress the array
    cmpr_data_bytes = compressor.compress(ori_data=ori_data)
    cmpr_data = np.frombuffer(cmpr_data_bytes, dtype=np.uint8)
    # Decompress the array
    dec_data = compressor.decompress(
        cmp_data=cmpr_data, ori_shape=ori_shape, ori_dtype=ori_dtype
    )
    # Check if the decompressed array is the same as the original array
    (max_diff, _, _) = compressor.verify(ori_data=ori_data, dec_data=dec_data)
    assert max_diff < cfg.compressor_error_bound


def test_model_compress(cfg: Config) -> None:
    # Create a compressor
    compressor = Compressor(cfg)

    # Create a model
    model = CNN(num_channel=1, num_classes=62, num_pixel=28)
    model_copy = copy.deepcopy(model)
    params = utils.flatten_model_params(model)
    ori_shape = params.shape
    ori_dtype = params.dtype

    # Compress the model
    cmpr_params_bytes = compressor.compress(ori_data=params)
    # Decompress the model
    dec_params = compressor.decompress(
        cmp_data=cmpr_params_bytes, ori_shape=ori_shape, ori_dtype=ori_dtype
    )
    # Check if the decompressed model is the same as the original model
    (max_diff, _, _) = compressor.verify(ori_data=params, dec_data=dec_params)
    assert max_diff < cfg.compressor_error_bound

    # Reasseble the model
    utils.unflatten_model_params(model, dec_params)
    # Check if the reassembled model is the same shape as the original model
    for p, p_copy in zip(model.parameters(), model_copy.parameters()):
        assert p.shape == p_copy.shape


if __name__ == "__main__":
    # Config setup
    cfg = OmegaConf.structured(Config)
    cfg.compressed_weights_client = True
    cfg.compressor = "SZ3"
    cfg.compressor_lib_path = "/Users/grantwilkins/SZ3/build/tools/sz3c/libSZ3c.dylib"
    # cfg.compressor_lib_path = "/Users/grantwilkins/SZ/build/sz/libSZ.dylib"
    cfg.compressor_error_bound = 0.1
    cfg.compressor_error_mode = "PW_REL"

    # Tests to run
    test_basic_compress(cfg=cfg)
    test_model_compress(cfg=cfg)
