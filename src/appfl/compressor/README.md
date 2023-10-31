# SZ Lossy Compressed Weights (Experimental)

The goal of this section is to lossy compress the model parameters or gradients before the client sends them back to the server for model updates. The server then will decompress the message and implement them using the algorithm specified by the user.

We implement the following lossy compressors:

1. [SZ2](https://github.com/szcompressor/SZ)
2. [SZ3](https://github.com/szcompressor/SZ3)
3. SZx
4. [ZFP](https://pypi.org/project/zfpy/)
