How to set configuration
========================

All runs use `OmegaConf <https://omegaconf.readthedocs.io/>`_, an YAML based hierarchical configuration system.
We define the configuration variables and their default values in YAML files located at ``src/appfl/config/*``.

We recommend to use `Hydra <https://hydra.cc>`_ that can read and load the configuration files to the ``OmegaConf.DictConfig`` object.
Please see the example codes in :ref:`How to simulate PPFL`.

Main configuration
------------------

.. literalinclude:: /../src/appfl/config/config.yaml
    :language: YAML
    :caption: Configuration file: src/appfl/config/config.yaml

This is the main configuration file ``config.yaml``.
Most keys are self-explanatory.

- Key ``defaults.fed`` sets the choice of algorithms. The value should be set as one of the file names (e.g., ``fedavg``, ``iceadmm``, ``iiadmm``).
- gRPC section sets the environment for running gRPC.
    - ``max_message_size``: the maximum size of data to be sent or received in a single RPC call, default 10 MB. If the size of weights for a single neuron is larger than 10 MB, you need to increase this value.
    - ``host``: the URL of a server
    - ``port``: the port number of a server


Algorithm configuration
-----------------------

Each algorithm, specified in ``defaults.fed``, can be configured in the files at ``config/fed``.
An example is given in the following:

.. literalinclude:: /../src/appfl/config/fed/fedavg.yaml
    :language: YAML
    :caption: Configuration file: src/appfl/config/fed/fedavg.yaml