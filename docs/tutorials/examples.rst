Running examples
================

To run the APPFL framework, the user-defined "model" and "datasets" should be prespecified.

- In this example, we use the following convolutional neural network model:

.. code-block:: python     

    class CNN(nn.Module):
        def __init__(self, num_channel, num_classes, num_pixel):
            super().__init__()
            self.conv1 = nn.Conv2d(
                num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
            )
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
            self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
            self.act = nn.ReLU(inplace=True)

            ###
            ### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
            ###
            X = num_pixel
            X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
            X = X / 2
            X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
            X = X / 2
            X = int(X)

            self.fc1 = nn.Linear(64 * X * X, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.act(self.conv1(x))
            x = self.maxpool(x)
            x = self.act(self.conv2(x))
            x = self.maxpool(x)
            x = torch.flatten(x, 1)
            x = self.act(self.fc1(x))
            x = self.fc2(x)
            return x


- We assume that the preprocessed datasets are stored in the ``datasets/PreprocessedData`` directory as explained in the ``Preprocessing datasets`` section. 
- In this example, we use the following preprocessed datasets:

  - ``datasets/PreprocessedData/MNIST_Clients_4``
  - ``datasets/PreprocessedData/CIFAR10_Clients_4``
  - ``datasets/PreprocessedData/Coronahack_Clients_4``
  - ``datasets/PreprocessedData/FEMNIST_Clients_203``

For each dataset, we construct ``mnist.py``, ``cifar10.py``, ``coronahack.py``, and ``femnist.py``, which are identical except:

.. code-block:: python     

    DataSet_name = "MNIST" 
    num_clients = 4
    num_channel = 1    
    num_classes = 10   
    num_pixel   = 28   

.. code-block:: python     

    DataSet_name = "CIFAR10" 
    num_clients = 4
    num_channel = 3    
    num_classes = 10   
    num_pixel   = 32       

.. code-block:: python     

    DataSet_name = "Coronahack" 
    num_clients = 4
    num_channel = 3    
    num_classes = 7   
    num_pixel   = 32  

.. code-block:: python     

    DataSet_name = "FEMNIST" 
    num_clients = 203
    num_channel = 1    
    num_classes = 62   
    num_pixel   = 28  

 