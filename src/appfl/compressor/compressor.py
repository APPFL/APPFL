from collections import OrderedDict
from copy import deepcopy
import zlib
from . import pysz
from ..config import Config
from typing import Tuple, Any
import numpy as np
import pickle
from . import pyszx
import zfpy
import scipy.sparse as sparse
import zstd
import torch.nn as nn
import torch
import gzip
import blosc
import lzma


class Compressor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sz_error_mode_dict = {
            "ABS": 0,
            "REL": 1,
            "ABS_AND_REL": 2,
            "ABS_OR_REL": 3,
            "PSNR": 4,
            "NORM": 5,
            "PW_REL": 10,
        }
        self.lossless_compressor = cfg.lossless_compressor
        self.compression_layers = []
        self.compressor_lib_path = ""
        self.param_count_threshold = cfg.param_cutoff
        if self.cfg.lossy_compressor == "SZ3":
            self.compressor_lib_path = self.cfg.compressor_sz3_path
        elif self.cfg.lossy_compressor == "SZ2":
            self.compressor_lib_path = self.cfg.compressor_sz2_path
        elif self.cfg.lossy_compressor == "SZx":
            self.compressor_lib_path = self.cfg.compressor_szx_path

    def compress(self, ori_data: np.ndarray):
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

    def compress_model(self, model: nn.Module) -> Tuple[bytes, int]:
        compressed_weights = {}
        lossy_compressed_size = 0
        lossy_original_size = 0
        lossy_elements = 0
        lossless_compressed_size = 0
        lossless_original_size = 0
        for name, param in model.items():
            param_flat = param.flatten().detach().cpu().numpy()
            if "weight" in name and param_flat.size > self.param_count_threshold:
                lossy_original_size += param_flat.size * 4
                lossy_elements += param_flat.size
                compressed_weights[name] = self.compress(ori_data=param_flat)
                lossy_compressed_size += len(compressed_weights[name])
            else:
                lossless_original_size += param_flat.size * 4
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
        if lossy_compressed_size != 0:
            print(
                "Lossy Compression Ratio: "
                + str(lossy_original_size / lossy_compressed_size)
            )
        print(
            "Total Compression Ratio: "
            + str(
                (lossy_original_size + lossless_original_size)
                / (lossy_compressed_size + lossless_compressed_size)
            )
        )
        print(
            "Lossless Compression Ratio: "
            + str(lossless_original_size / lossless_compressed_size)
        )
        return (
            pickle.dumps(compressed_weights),
            lossy_elements,
        )

    def decompress_model(self, compressed_model: bytes, model: nn.Module) -> nn.Module:
        model_copy = deepcopy(model)
        new_dict = OrderedDict()
        decomp_weights = pickle.loads(compressed_model)
        for name, param in model_copy.state_dict().items():
            if "weight" in name and param.numel() > self.param_count_threshold:
                decomp_weights[name] = self.decompress(
                    cmp_data=decomp_weights[name],
                    ori_shape=(param.numel(),),
                    ori_dtype=np.float32,
                )
            else:
                if self.lossless_compressor == "zstd":
                    decomp_weights[name] = zstd.decompress(decomp_weights[name])
                elif self.lossless_compressor == "gzip":
                    decomp_weights[name] = gzip.decompress(decomp_weights[name])
                elif self.lossless_compressor == "zlib":
                    decomp_weights[name] = zlib.decompress(decomp_weights[name])
                elif self.lossless_compressor == "blosc":
                    decomp_weights[name] = blosc.decompress(
                        decomp_weights[name], as_bytearray=True
                    )
                elif self.lossless_compressor == "lzma":
                    decomp_weights[name] = lzma.decompress(decomp_weights[name])
                else:
                    raise NotImplementedError
                decomp_weights[name] = np.frombuffer(
                    decomp_weights[name], dtype=np.float32
                )
            if param.shape == torch.Size([]):
                copy_arr = deepcopy(decomp_weights[name])
                copy_tensor = torch.from_numpy(copy_arr)
                new_dict[name] = torch.tensor(copy_tensor)
            else:
                copy_arr = deepcopy(decomp_weights[name])
                copy_tensor = torch.from_numpy(copy_arr)
                new_dict[name] = copy_tensor.reshape(param.shape)
        # model_copy.load_state_dict(new_dict)
        # return model_copy
        return new_dict

    def decompress(
        self, cmp_data, ori_shape: Tuple[int, ...], ori_dtype: np.dtype
    ) -> np.ndarray:
        # Decompress data with chosen compressor
        # :param cmp_data: compressed data, numpy array format, dtype should be np.uint8
        # :param ori_shape: the shape of original data
        # :param ori_dtype: the dtype of original data
        # :return: decompressed data,numpy array format
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
