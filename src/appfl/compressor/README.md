# SZ Lossy Compressed Weights (Experimental)

The goal of this section is to lossy compress the model parameters or gradients before the client sends them back to the server for model updates. The server then will decompress the message and implement them using the algorithm specified by the user.

We implement the SZ lossy compressor with versions:

1. [SZ2](https://github.com/szcompressor/SZ)
2. [SZ3](https://github.com/szcompressor/SZ3)
3. SZx

Simply add the config options:

```python
cfg.compressed_weights = True
cfg.compressor = SZ2 or SZ3 or SZx
cfg.compressor_lib = /path/to/dynamic/libary
cfg.compressor_error_mode = ABS or REL or PW_REL
cfg.compressor_error_bound = err_bound
```

where `err_bound` is a floating point error bound.
