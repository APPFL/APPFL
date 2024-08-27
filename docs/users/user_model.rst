How to define model
===================

User-defined models can be anything derived from ``torch.nn.Module`` with any keyword arguments necessay.
For example, we can define a fully connected (FC) layer as follows:

.. code-block:: python

    class FC(nn.Module):
        """
        A Fully connected layer.
        """

        def __init__(self, input_size):
            super(FC, self).__init__()
            self.fc = nn.Linear(input_size, 1)

        def forward(self, x):
            out = self.fc(x)
            return out 

To use the model, you need to provide the absolute/relative path to the model definition file, the name of the model class, and the keyword arguments to the model class. For example, to use the FC layer defined above, you can add the following lines to the server configuration file:

.. code-block:: yaml
    
    client_configs:
        ...
        model_configs:
            model_path: "<path_to_fc>.py"
            model_name: "FC"
            model_kwargs:
                input_size: 39
        ...