How to train a custom model
===========================

To run our APPFL framework, *user-defined* dataset and model are required. In this section we explain how to run our APPFL framework using **FEMNIST** dataset and **Convolutional Neural Network (CNN)** model. 

Prerequisites
-------------

Download datasets
^^^^^^^^^^^^^^^^^

- Git clone ``https://github.com/TalwalkarLab/leaf.git``

- In a ``leaf/data/femnist`` directory, do ``./preprocess.sh -s niid --sf 0.05 -k 0 -t sample`` which downloads **a small-sized FEMNIST dataset**.

    - # of training data is varied over clients (# of clients = 203)
    - # of testing data = 4176 
    - # of classes = 62
    - # of channels = 1       
    - # of pixels = 28 * 28

- In a newly generated directory ``leaf/data/femnist/data``, copy the two directories ``train`` and ``test`` and paste them in a ``datasets/RawData/FEMNIST`` directory.
    

Creating a custom dataset for your data
---------------------------------------

Our APPFL framework takes a dataset expressed by ``torch`` tensors.

Creating a custom model
-----------------------

The users of our APPFL framework can create a machine learning model trained by multiple clients. See ``examples/models/cnn.py`` as an example, a CNN model with the following structure:

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

Run
---

To run our APPFL framework using ``FEMNIST`` and ``CNN``, the users should create a python file like a ``examples/femnist.py``.

1. Read the dataset

.. code-block:: python     

    DataSet_name = "FEMNIST" 
    num_clients = 203
    num_channel = 1    # 1 if gray, 3 if color
    num_classes = 62   # number of the image classes 
    num_pixel   = 28   # image size = (num_pixel, num_pixel)

    dir = "../datasets/RawData/%s" % (DataSet_name)

    # test data for a server
    test_data_raw = {}
    test_data_input = []
    test_data_label = []
    for idx in range(36):
        with open("%s/test/all_data_%s_niid_05_keep_0_test_9.json" % (dir, idx)) as f:
            test_data_raw[idx] = json.load(f)

        for client in test_data_raw[idx]["users"]:

            for data_input in test_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(28, 28)
                test_data_input.append([data_input])

            for data_label in test_data_raw[idx]["user_data"][client]["y"]:
                test_data_label.append(data_label)

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = {}
    train_datasets = []
    for idx in range(36):
        with open("%s/train/all_data_%s_niid_05_keep_0_train_9.json" % (dir, idx)) as f:
            train_data_raw[idx] = json.load(f)

        for client in train_data_raw[idx]["users"]:

            train_data_input_resize = []
            for data_input in train_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(28, 28)
                train_data_input_resize.append([data_input])

            train_datasets.append(
                Dataset(
                    torch.FloatTensor(train_data_input_resize),
                    torch.tensor(train_data_raw[idx]["user_data"][client]["y"]),
                )
            )

    data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)


2. Load the model

.. code-block:: python     

    model = CNN(num_channel, num_classes, num_pixel)

3. Run

.. code-block:: python     

    @hydra.main(config_path="../appfl/config", config_name="config")
    def main(cfg: DictConfig):
        
        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
    
        torch.manual_seed(1)
    
        if comm_size > 1:
            if comm_rank == 0:            
                rt.run_server(cfg, comm, model, test_dataset, num_clients, DataSet_name)
            else:                        
                rt.run_client(cfg, comm, model, train_datasets, num_clients)
            print("------DONE------", comm_rank)
        else:
            rt.run_serial(cfg, model, train_datasets, test_dataset)


In the ``examples`` directory, do the followings:

# To run ``run_serial`` using a CPU:

>>> python ./femnist.py device=cpu fed=fedavg num_epochs=10 fed.args.num_local_epochs=5 

# To run ``run_serial`` using a GPU:

>>> python ./femnist.py device=cuda fed=fedavg num_epochs=10 fed.args.num_local_epochs=5 

# To run ``run_server`` using a CPU and ``run_client`` using 4 CPUs:

>>> mpiexec -np 5 python ./femnist.py device=cpu fed=fedavg num_epochs=10 fed.args.num_local_epochs=5

# To run ``run_server`` using a GPU and ``run_client`` using 4 GPUs:

>>> mpiexec -np 5 --mca opal_cuda_support 1 python ./femnist.py device=cuda fed=fedavg num_epochs=10 fed.args.num_local_epochs=5


Note that the above commands run the APPFL framework by using the Federated Averaging (FedAvg) algorithm with 10 communication rounds and 5 local updates.

To use IADMM as an algorithm, change ``fed=fedavg`` to ``fed=iadmm`` and add ``fed.args.penalty=100.0``, where "100" is the ADMM penalty parameter which should be fine-tuned.

