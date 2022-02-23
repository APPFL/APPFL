Loading and saving models
===========================

Loading
-------
Pretrained ML models in ``TorchScript`` format can be loaded in APPFL.
See https://pytorch.org/tutorials/beginner/saving_loading_models.html for more details on the ``TorchScript`` format

The model parameters obtained from the pretrained model can be used as an initial guess of the FL model parameters in APPFL.
To do this, one should revise the configuration file (i.e., ``src/appfl/config/config.py``).
For example, suppose that we have a pretrained model stored as ``model.pt`` in a ``examples/models`` directory.
Then, one can revise the configuration file as follows:

.. code-block:: python
    
    # Loading Models
    load_model: bool = True
    load_model_dirname: str = "./models"
    load_model_filename: str = "model"

Saving
------
After federated learning, the resulting models can be stored in ``TorchScript`` format.
To do this, one should revise the configuration file. See the followings for an example:

.. code-block:: python

    # Saving Models
    save_model: bool = True
    save_model_dirname: str = "./save_models"
    save_model_filename: str = "model"
