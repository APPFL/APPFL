User-defined model
=====================

User-defined models can be anything derived from ``torch.nn.Module``.
For example, we can define a convolutional neural network model as follows:

.. literalinclude:: /../examples/models/cnn.py
    :language: python
    :caption: User-defined convolutionsl neural network model: examples/models/cnn.py

In the code, one can add the CNN model and the loss function as follows:

.. code-block:: python

    model = CNN()
    loss_fn = torch.nn.CrossEntropyLoss()   

Note that the loss_fn can be anything derived from ``torch.nn.Module``.