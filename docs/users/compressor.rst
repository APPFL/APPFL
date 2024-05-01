APPFL Compressor
================

Currently, APPFL supports the following lossy compressors:

- `SZ2: Error-bounded Lossy Compressor for HPC Data <https://github.com/szcompressor/SZ>`_
- `SZ3: A Modular Error-bounded Lossy Compression Framework for Scientific Datasets <https://github.com/szcompressor/SZ3>`_
- `SZx: An Ultra-fast Error-bounded Lossy Compressor for Scientific Datasets <https://github.com/szcompressor/SZx>`_
- `ZFP: Compressed Floating-Point and Integer Arrays <https://pypi.org/project/zfpy/>`_

Installation
------------

User can install the compressors by running the following command:

.. code-block:: bash

    appfl-install-compressor

.. note::
    SZx is not open source so we omit its installation here. Please install it manually by contacting the author.

Configuration
-------------
User can configure the compressor by setting it `client_configs.comm_configs.compressor_configs` in the server configuration file. The following is an example of the configuration:

.. code-block:: python

    client_configs:
        comm_configs:
            compressor_configs:
            enable_compression: True
            lossy_compressor:  "SZ2"
            lossless_compressor: "blosc"
            error_bounding_mode: "REL"
            error_bound: 1e-3
            flat_model_dtype: "np.float32"
            param_cutoff: 1024

Usage
-----

The compressor is used in the `APPFLClientAgent.get_parameters` method to compress the model parameters using `compressor.compress_model` before sending them to the server, as shown below

.. code-block:: python

    def get_parameters(self) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
        """Return parameters for communication"""
        params = self.trainer.get_parameters()
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        if self.enable_compression:
            params = self.compressor.compress_model(params)
        return params if metadata is None else (params, metadata)

On the server side, the model parameters are decompressed using `compressor.decompress_model` before updating the model by the `APPFLServerAgent.global_update`.