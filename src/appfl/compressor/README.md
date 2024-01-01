# 🗜 Model Parameter Compressor

The compressor is used for compressing the model parameters or gradients in a lossy manner before the client sends them back to the server for communication efficiency. The server then will decompress the compressed model for aggregation. 

We implement the following lossy compressors. Please refer to their official project/GitHub pages if you want more detailed information of them. Here, we only provide the installation instructions. **Note: SZx need particular permission to access because collaboration with a third-party, so we omit its installation here.**.

1. [SZ2](https://github.com/szcompressor/SZ)
2. [SZ3](https://github.com/szcompressor/SZ3)
3. [ZFP](https://pypi.org/project/zfpy/)
4. [SZx](https://github.com/szcompressor/SZx)

## Installation
We provide a script for installing all the above compressors [`examples/compressor/install.sh`](../../../examples/compressor/install.sh). You can install them by running the following commands in the `examples` directory.

```bash
cd examples
chmod +x compressor/install.sh
./compressor/install.sh
```