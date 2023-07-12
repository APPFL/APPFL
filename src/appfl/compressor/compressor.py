from . import pysz
from ..config import Config
from typing import Tuple, Any
import numpy as np


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
        self.compressor_class = None

    def compress(self, ori_data: np.ndarray):
        """
        Compress data with chosen compressor
        :param ori_data: compressed data, numpy array format
        :return: decompressed data,numpy array format
        """
        if self.cfg.compressor == "SZ3" or "SZ2":
            self.cfg.flat_model_size = ori_data.shape
            if self.compressor_class is None:
                self.compressor_class = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[self.cfg.compressor_error_mode]
            error_bound = self.cfg.compressor_error_bound
            compressed_arr, comp_ratio = self.compressor_class.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
                eb_pwr=error_bound,
            )
            return compressed_arr.tobytes()
        else:
            raise NotImplementedError

    def decompress(
        self, cmp_data, ori_shape: Tuple[int, ...], ori_dtype: np.dtype
    ) -> np.ndarray:
        """
        Decompress data with chosen compressor
        :param cmp_data: compressed data, numpy array format, dtype should be np.uint8
        :param ori_shape: the shape of original data
        :param ori_dtype: the dtype of original data
        :return: decompressed data,numpy array format
        """
        if self.cfg.compressor == "SZ3" or "SZ2":
            if self.compressor_class is None:
                self.compressor_class = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            cmp_data = np.frombuffer(cmp_data, dtype=np.uint8)
            decompressed_arr = self.compressor_class.decompress(
                data_cmpr=cmp_data,
                original_shape=ori_shape,
                original_dtype=ori_dtype,
            )
            return decompressed_arr
        else:
            raise NotImplementedError

    def verify(self, ori_data, dec_data) -> Tuple[float, ...]:
        if self.cfg.compressor == "SZ3" or "SZ2":
            if self.compressor_class is None:
                self.compressor_class = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            return self.compressor_class.verify(ori_data, dec_data)
        else:
            raise NotImplementedError
