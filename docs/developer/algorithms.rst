How to add new algorithms
=========================

Suppose that we are adding the configuration for our new algorithm.
New algorithm should be implemented as two classes for server and client. 
Implementation of the new classes should be derived from the following two base classes:

.. autoclass:: appfl.algorithm.BaseServer
    :members:

.. autoclass:: appfl.algorithm.BaseClient
    :members:

Example: NewAlgo
----------------

Here we give some simple example.

Core algorithm class
++++++++++++++++++++

We first create classes for the global and local updates in ``appfl/algorithm``:

- See two classes ``NewAlgoServer`` and ``NewAlgoClient`` in ``newalgo.py``
- In ``NewAlgoServer``, the ``update`` function conducts a global update by averaging the local model parameters sent from multiple clients
- In ``NewAlgoClient``, the ``update`` function conducts a local update and send the resulting local model parameters to the server

This is an example code:

.. code-block:: python
    :caption: Example code for ``src/appfl/algorithm/newalgo.py``

    from .algorithm import BaseServer, BaseClient

    class NewAlgoServer(BaseServer):
        def __init__(self, weights, model, num_clients, device, **kwargs):
            super(NewAlgoServer, self).__init__(weights, model, num_clients, device)
            self.__dict__.update(kwargs)
            # Any additional initialization

        def update(self, local_states: OrderedDict):
            # Implement new server update function

    class NewAlgoClient(BaseClient):
        def __init__(self, id, weight, model, dataloader, device, **kwargs):
            super(NewAlgoClient, self).__init__(id, weight, model, dataloader, device)
            self.__dict__.update(kwargs)
            # Any additional initialization

        def update(self):
            # Implement new client update function


Configuration dataclass
+++++++++++++++++++++++

The new algorithm also needs to set up some configurations. This can be done by adding new dataclass under ``appfl.config.fed``.
Let's say we add ``src/appfl/config/fed/newalgo.py`` file to implement the dataclass as follows:

.. code-block:: python
    :caption: Example code for ``src/appfl/config/fed/newalgo.py``

    from dataclasses import dataclass
    from omegaconf import DictConfig, OmegaConf

    @dataclass
    class NewAlgo:
        type: str = "newalgo"
        servername: str = "NewAlgoServer"
        clientname: str = "NewAlgoClient"
        args: DictConfig = OmegaConf.create(
            {
                # add new arguments
            }
        )


Then, we need to add the following line to the main configuration file ``config.py``.

.. code-block:: python

    from .fed.new_algorithm import *


This is the main configuration class in ``src/appfl/config/config.py``.
Each algorithm, specified in ``Config.fed``, can be configured in the dataclasses at ``appfl.config.fed.*``.

.. literalinclude:: /../src/appfl/config/config.py
    :language: python
    :linenos:
    :caption: The main configuration class
