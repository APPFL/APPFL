Loading and saving models
===========================

Loading
-------
Pretrained ML models can be loaded in APPFL via ``torch.load()``.
See https://pytorch.org/tutorials/beginner/saving_loading_models.html for more details.

The model parameters obtained from the pretrained model can be used as an initial guess of the FL model parameters in APPFL.
To use this feature, one should revise the configuration file (i.e., ``src/appfl/config/config.py``).
For example, suppose that we have a pretrained model stored as ``model.pt`` in a ``examples/models`` directory.
Then, one can revise the configuration file as follows:

.. code-block:: python
    
    # Loading Models
    load_model: bool = True
    load_model_dirname: str = "./models"
    load_model_filename: str = "model"

Saving
------
After federated learning, the resulting models can be stored via ``torch.save()``.
To use this feature, one should revise the configuration file. See the followings for an example:

.. code-block:: python

    # Saving Models
    save_model: bool = True
    save_model_dirname: str = "./save_models"
    save_model_filename: str = "model"
    checkpoints_interval: int = 2

By setting ``checkpoints_interval = 2``, trained model will be saved for every 2 iteration.

Note: When using ``docker container``, one can download the trained models via ``docker cp`` (https://docs.docker.com/engine/reference/commandline/cp/)

For example, if my container ID is ``aa90d20f96c0d143012d2e6ca7d7820ed9ed8a36b163cddf8bfd6dd0e6228dab`` then

.. code-block:: python

    docker cp aa90d20f96c0d143012d2e6ca7d7820ed9ed8a36b163cddf8bfd6dd0e6228dab:/APPFL/save_models/ .