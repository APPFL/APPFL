How to load datasets
====================

Each client loads its local private dataset by providing a function that returns the training and validation datasets as ``torch.utils.data.Dataset`` objects. ``APPFL`` provides a simple dataset class, ``appfl.misc.data.Dataset``, which takes ``data_input`` and ``data_label`` as ``torch.Tensor`` objects. We expect this simple class to be sufficient in most cases. However, users can also create more sophisticated dataset classes for their own needs.

For example, suppose that we define the following function to load the dataset:

.. code-block:: python

    def get_my_local_dataset(
        **kwargs
    ):
        ...
        return train_dataset, val_dataset

Then we can load the dataset by providing the absolute/relative path to the function definition file, the function name, and the keyword arguments in the client configuration file as follows:

.. code-block:: yaml

    # Local dataset
    data_configs:
      dataset_path: <path_to_your_dataset_fn>.py
      dataset_name: "get_my_local_dataset"
      dataset_kwargs:
        ...
