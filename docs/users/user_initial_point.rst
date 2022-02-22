Initial point to start with
===========================

To warm start APPFL, one can incorporate a pretrained model into APPFL.
The pretrained model should be written in ``TorchScript`` format (for more details, see https://pytorch.org/tutorials/beginner/saving_loading_models.html). 

As an example, suppose that we have a pretrained model stored as ``mnist_cnn_initial.pt`` in a ``examples/models`` directory.
To use the model as an initial point to start APPFL with, one can set the followings in ``src/appfl/config/config.py``:

.. code-block:: python
    
    # Initial Model Parameters
    is_init_point: bool = True
    init_point_dir: str = "./models"
    init_point_filename: str = "mnist_cnn_initial.pt"
