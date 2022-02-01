Examples
========

This describes the examples given in ``examples`` directory.


User-defined model
------------------

User-defined models can be anything derived from ``torch.nn.Module``.
In this example, we use the following convolutional neural network model:

.. literalinclude:: /../examples/models/cnn.py
    :language: python
    :caption: User-defined convolutionsl neural network model: examples/models/cnn.py


User-defined dataset
--------------------

We assume that the preprocessed datasets are stored in the ``datasets/PreprocessedData`` directory as explained in the ``Preprocessing datasets`` section. 
In this example, we use the following preprocessed datasets:

  - ``datasets/PreprocessedData/MNIST_Clients_4``
  - ``datasets/PreprocessedData/CIFAR10_Clients_4``
  - ``datasets/PreprocessedData/Coronahack_Clients_4``
  - ``datasets/PreprocessedData/FEMNIST_Clients_203``

For each dataset, we construct ``mnist.py``, ``cifar10.py``, ``coronahack.py``, and ``femnist.py``, which are identical except:

.. list-table:: Parameter settings for each dataset in the examples
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Example file
      - num_clients
      - num_channel
      - num_classes
      - num_pixel
    * - ``mnist.py``
      - 4
      - 1
      - 10
      - 28
    * - ``cifar10.py``
      - 4
      - 3
      - 10
      - 32
    * - ``coronahack.py``
      - 4
      - 3
      - 7
      - 32
    * - ``femnist.py``
      - 203
      - 1
      - 62
      - 28
 

.. toctree::
   :maxdepth: 1
   :titlesonly:
   
   datasets