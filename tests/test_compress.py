from appfl.compressor. compressor import *
import numpy as np
from appfl.config import *

def main():
    cfg = OmegaConf.structured(Config)
    cfg.compressor = "SZ3"
    cfg.compressor_lib_path = "/Users/grantwilkins/SZ3/build/tools/sz3c/libSZ3c.dylib"
    cfg.compressor_error_bound = 0.1
    cfg.compressor_error_mode = "ABS"
    compressor = Compressor(cfg)
    # Create a random 1D array
    ori_data = np.random.rand(1000)
    ori_shape = ori_data.shape
    ori_dtype = ori_data.dtype
    # Compress the array
    cmpr_data, cmpr_ratio = compressor.compress(ori_data=ori_data)
    # Decompress the array
    dec_data = compressor.decompress(cmp_data=cmpr_data, 
                                     ori_shape=ori_shape, 
                                     ori_dtype=ori_dtype)
    # Check if the decompressed array is the same as the original array
    (max_diff, psnr, nsmre) = compressor.verify(ori_data=ori_data, dec_data=dec_data)
    assert max_diff < cfg.compressor_error_bound

if __name__ == "__main__":
    main()