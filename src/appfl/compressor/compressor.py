import os
import sys
import gzip
import lzma
import zfpy
import zlib
import zstd
import blosc
import torch
import pickle
import numpy as np
from . import pysz
from . import pyszx
from copy import deepcopy
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Tuple, Union, List

class Compressor:
    def __init__(self, compressor_config: DictConfig):
        current_path = os.path.dirname(os.path.abspath(__file__))
        appfl_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
        self.cfg = compressor_config
        self.sz_error_mode_dict = {
            "ABS": 0,
            "REL": 1,
            "ABS_AND_REL": 2,
            "ABS_OR_REL": 3,
            "PSNR": 4,
            "NORM": 5,
            "PW_REL": 10,
        }
        self.lossless_compressor = compressor_config.lossless_compressor
        self.compression_layers = []
        self.compressor_lib_path = ""
        self.param_count_threshold = compressor_config.param_cutoff
        ext = ".dylib" if sys.platform.startswith("darwin") else ".so"
        if self.cfg.lossy_compressor == "SZ3":
            self.compressor_lib_path = os.path.join(appfl_root_dir, ".compressor/SZ/build/sz/libSZ") + ext
        elif self.cfg.lossy_compressor == "SZ2":
            self.compressor_lib_path = os.path.join(appfl_root_dir, ".compressor/SZ3/build/tools/sz3c/libSZ3c") + ext
        elif self.cfg.lossy_compressor == "SZx":
            self.compressor_lib_path = os.path.join(appfl_root_dir, ".compressor/SZx-main/build/lib/libSZx") + ext

    def compress_model(
        self, 
        model: Union[dict, OrderedDict, List[Union[dict, OrderedDict]]], 
        batched: bool=False
    ) -> bytes:
        """
        Compress all the parameters of local model(s) for efficient communication. The local model can be batched as a list.
        :param model: local model parameters (can be nested)
        :param batched: whether the input is a batch of models
        :return: compressed model parameters as bytes
        """
        # Deal with batched models
        if batched:
            if isinstance(model, list):
                compressed_models = []
                num_lossy_elements = 0
                for model_sample in model:
                    compressed_model = self.compress_model(model_sample)
                    compressed_models.append(compressed_model)
                    num_lossy_elements += lossy_elements
                return pickle.dumps(compressed_models)
            if isinstance(model, dict) or isinstance(model, OrderedDict):
                compressed_models = OrderedDict()
                num_lossy_elements = 0
                for key, model_sample in model.items():
                    compressed_model = self.compress_model(model_sample)
                    compressed_models[key] = compressed_model
                    num_lossy_elements += lossy_elements
                return pickle.dumps(compressed_models)

        for _, value in model.items():
            is_nested = not isinstance(value, torch.Tensor)
            break
        
        if is_nested:
            num_lossy_elements = 0
            compressed_models = OrderedDict()
            for key, weights in model.items():
                if isinstance(weights, dict) or isinstance(weights, OrderedDict):
                    comprsessed_weights, lossy_elements = self._compress_weights(weights)
                    compressed_models[key] = comprsessed_weights
                    lossy_elements += lossy_elements
                else:
                    compressed_models[key] = weights
        else:
            compressed_models, num_lossy_elements = self._compress_weights(model)
        return pickle.dumps(compressed_models)

    def decompress_model(
        self, 
        compressed_model: bytes, 
        model: Union[dict, OrderedDict], 
        batched: bool=False
    )-> Union[OrderedDict, dict, List[Union[OrderedDict, dict]]]:
        """
        Decompress all the communicated model parameters. The local model can be batched as a list.
        :param compressed_model: compressed model parameters as bytes
        :param model: a model sample for de-compression reference
        :param batched: whether the input is a batch of models
        :return decompressed_model: decompressed model parameters
        """
        compressed_model = pickle.loads(compressed_model)
        
        # Deal with batched models
        if batched:
            if isinstance(compressed_model, list):
                decompressed_models = []
                for compressed_model_sample in compressed_model:
                    decompressed_model_sample = self.decompress_model(compressed_model_sample, model)
                    decompressed_models.append(decompressed_model_sample)
                return decompressed_models
            if isinstance(compressed_model, dict) or isinstance(compressed_model, OrderedDict):
                decompressed_models = OrderedDict()
                for key, compressed_model_sample in compressed_model.items():
                    decompressed_model_sample = self.decompress_model(compressed_model_sample, model)
                    decompressed_models[key] = decompressed_model_sample
                return decompressed_models

        for _, value in compressed_model.items():
            is_nested = not isinstance(value, bytes)
            break
        if is_nested:
            decompressed_model = OrderedDict()
            for key, value in compressed_model.items():
                if isinstance(value, dict) or isinstance(value, OrderedDict):
                    decompressed_model[key] = self._decompress_model(value, model)
                else:    
                    decompressed_model[key] = value
        else:
            decompressed_model = self._decompress_model(compressed_model, model)
        return decompressed_model

    def _compress_weights(
        self, 
        weights: Union[OrderedDict, dict]
    ) -> Tuple[Union[OrderedDict, dict], int]:
        """
        Compress ONE set of weights of the model.
        :param weights: the model weights to be compressed
        :return: the compressed model weights and the number of lossy elements
        """
        # Check if the input a set of model weights
        if len(weights) == 0:
            return (weights, 0)
        for _, value in weights.items():
            if not isinstance(value, torch.Tensor):
                return (weights, 0)
            break

        compressed_weights = {}
        lossy_elements = 0
        lossy_original_size = 0
        lossy_compressed_size = 0
        lossless_original_size = 0
        lossless_compressed_size = 0

        for name, param in weights.items():
            param_flat = param.flatten().detach().cpu().numpy()
            if "weight" in name and param_flat.size > self.param_count_threshold:
                lossy_original_size += param_flat.nbytes
                lossy_elements += param_flat.size
                compressed_weights[name] = self._compress(ori_data=param_flat)
                lossy_compressed_size += len(compressed_weights[name])
            else:
                lossless_original_size += param_flat.nbytes
                lossless = b""
                if self.lossless_compressor == "zstd":
                    lossless = zstd.compress(param_flat, 10)
                elif self.lossless_compressor == "gzip":
                    lossless = gzip.compress(param_flat.tobytes())
                elif self.lossless_compressor == "zlib":
                    lossless = zlib.compress(param_flat.tobytes())
                elif self.lossless_compressor == "blosc":
                    lossless = blosc.compress(param_flat.tobytes(), typesize=4)
                elif self.lossless_compressor == "lzma":
                    lossless = lzma.compress(param_flat.tobytes())
                else:
                    raise NotImplementedError
                lossless_compressed_size += len(lossless)
                compressed_weights[name] = lossless
        # if lossy_compressed_size != 0:
        #     print("Lossy Compression Ratio: " + str(lossy_original_size / lossy_compressed_size))
        # if lossless_compressed_size != 0:
        #     print("Lossless Compression Ratio: " + str(lossless_original_size / lossless_compressed_size))
        # print("Total Compression Ratio: " + str((lossy_original_size + lossless_original_size) / (lossy_compressed_size + lossless_compressed_size)))
        return (
            compressed_weights,
            lossy_elements,
        )

    def _compress(self, ori_data: np.ndarray):
        """
        Compress data with chosen compressor
        :param ori_data: compressed data, numpy array format
        :return: decompressed data,numpy array format
        """
        if self.cfg.lossy_compressor == "SZ3" or self.cfg.lossy_compressor == "SZ2":
            compressor = pysz.SZ(szpath=self.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[self.cfg.error_bounding_mode]
            error_bound = self.cfg.error_bound
            compressed_arr, comp_ratio = compressor.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
                eb_pwr=error_bound,
            )
            return compressed_arr.tobytes()
        elif self.cfg.lossy_compressor == "SZx":
            compressor = pyszx.SZx(szxpath=self.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[self.cfg.error_bounding_mode]
            error_bound = self.cfg.error_bound
            compressed_arr, comp_ratio = compressor.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
            )
            return compressed_arr.tobytes()
        elif self.cfg.lossy_compressor == "ZFP":
            if self.cfg.error_bounding_mode == "ABS":
                return zfpy.compress_numpy(ori_data, tolerance=self.cfg.error_bound)
            elif self.cfg.error_bounding_mode == "REL":
                range_data = abs(np.max(ori_data) - np.min(ori_data))
                return zfpy.compress_numpy(
                    ori_data, tolerance=self.cfg.error_bound * range_data
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _decompress_model(
        self, 
        compressed_weights: Union[dict, OrderedDict], 
        model: Union[dict, OrderedDict]
    ) -> Union[OrderedDict, dict]:
        """
        Decompress ONE set of weights of the model.
        :param compressed_weights: the compressed model weights
        :param model: a model sample for de-compression reference
        :return: decompressed model weights
        """
        if len(compressed_weights) == 0:
            return compressed_weights
        for _, value in compressed_weights.items():
            if not isinstance(value, bytes):
                return compressed_weights
            break
        decompressed_weights = OrderedDict()
        for name, param in model.state_dict().items():
            if "weight" in name and param.numel() > self.param_count_threshold:
                compressed_weights[name] = self._decompress(
                    cmp_data=compressed_weights[name],
                    ori_shape=(param.numel(),),
                    ori_dtype=np.float32,
                ).astype(np.float32)                
            else:
                if self.lossless_compressor == "zstd":
                    compressed_weights[name] = zstd.decompress(compressed_weights[name])
                elif self.lossless_compressor == "gzip":
                    compressed_weights[name] = gzip.decompress(compressed_weights[name])
                elif self.lossless_compressor == "zlib":
                    compressed_weights[name] = zlib.decompress(compressed_weights[name])
                elif self.lossless_compressor == "blosc":
                    compressed_weights[name] = blosc.decompress(
                        compressed_weights[name], as_bytearray=True
                    )
                elif self.lossless_compressor == "lzma":
                    compressed_weights[name] = lzma.decompress(compressed_weights[name])
                else:
                    raise NotImplementedError
                compressed_weights[name] = np.frombuffer(
                    compressed_weights[name], dtype=np.float32
                )
            if param.shape == torch.Size([]):
                copy_arr = deepcopy(compressed_weights[name])
                copy_tensor = torch.from_numpy(copy_arr)
                decompressed_weights[name] = torch.tensor(copy_tensor)
            else:
                copy_arr = deepcopy(compressed_weights[name])
                copy_tensor = torch.from_numpy(copy_arr)
                decompressed_weights[name] = copy_tensor.reshape(param.shape)
        return decompressed_weights

    def _decompress(
        self, 
        cmp_data, 
        ori_shape: Tuple[int, ...], 
        ori_dtype: np.dtype
    ) -> np.ndarray:
        """
        Decompress data with chosen compressor
        :param cmp_data: compressed data, numpy array format, dtype should be np.uint8
        :param ori_shape: the shape of original data
        :param ori_dtype: the dtype of original data
        :return: decompressed data,numpy array format
        """
        if self.cfg.lossy_compressor == "SZ3" or self.cfg.lossy_compressor == "SZ2":
            compressor = pysz.SZ(szpath=self.compressor_lib_path)
            cmp_data = np.frombuffer(cmp_data, dtype=np.uint8)
            decompressed_arr = compressor.decompress(
                data_cmpr=cmp_data,
                original_shape=ori_shape,
                original_dtype=ori_dtype,
            )
            return decompressed_arr
        elif self.cfg.lossy_compressor == "SZx":
            compressor = pyszx.SZx(szxpath=self.compressor_lib_path)
            cmp_data = np.frombuffer(cmp_data, dtype=np.uint8)
            decompressed_arr = compressor.decompress(
                data_cmpr=cmp_data,
                original_shape=ori_shape,
                original_dtype=ori_dtype,
            )
            return decompressed_arr
        elif self.cfg.lossy_compressor == "ZFP":
            return zfpy.decompress_numpy(cmp_data)
        else:
            raise NotImplementedError
