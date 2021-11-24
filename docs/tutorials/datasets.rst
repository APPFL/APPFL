Preprocessing Datasets 
======================

In a federated learning setting, a server utilizes a **testing dataset** to evaluate a global model parameter contributed by multiple clients, each of which conducts learning using its own **training dataset**.

The APPFL framework reads datasets in a ``json`` format.

- In a ``datasets/PreprocessedData/[DatasetName]_Clients_[#Clients]`` directory, we store datasets with the following form: 

    - Testing dataset for **a server**: ``all_test_data.json`` 
    - Training dataset for **a client "i"**: ``all_train_data_client_${i}.json`` 

    Note: each ``json`` file has a form of ``{"x": [ a list of data inputs ], "y": [ a list of data labels ] }``.

To obtain the datasets with the above form, preprocessing raw datasets is required.

- In a ``datasets/RawData`` directory, store the raw datasets.
- Construct ``[DatasetName]_Preprocess.py`` that converts "the raw datasets" to "the preprocessed datasets"

**Example 1.** ``datasets/MNIST_Preprocess.py``

1. This code downloads MNIST datasets from TorchVision and store them in ``datasets/RawData``. 
   
   - # training data = 60000
   - # testing data = 10000 
   - # classes = 10
   - # channels = 1 
   - # pixels = 28

2. In the code, set ``num_clients``. 

   - As an example, we set  ``num_clients = 4``
   - This will divide the training datasets by 4 clients. Each client has 15000 training data points

3. This code generates ``json`` files in the ``datasets/PreprocessedData/MNIST_Clients_4`` directory.

**Example 2.** ``datasets/Coronahack_Preprocess.py``

1. Prerequisite:
 
   - ``mkdir Coronahack`` in ``datasets/RawData``
   - Download the dataset from https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset 
   - Place the dataset (i.e., "archive" directory) to the ``datasets/RawData/Coronahack`` directory
   - # training data = 5286
   - # testing data = 624 
   - # classes = 7
   - # channels = 3

2. In the code, we set ``num_clients = 4`` as an example. Additionally, we set ``num_pixel=32``.
3. This code generates ``json`` files in the ``datasets/PreprocessedData/Coronahack_Clients_4`` directory.

**Example 3.** ``datasets/FEMNIST_Preprocess.py``

1. Prerequisite:
 
   - ``mkdir FEMNIST`` in ``datasets/RawData``
   - Git clone https://github.com/TalwalkarLab/leaf.git
   - As an example, in ``leaf/data/femnist``, do ``./preprocess.sh -s niid --sf 0.05 -k 0 -t sample`` which downloads **a small-sized dataset**
   - In a newly generated directory ``leaf/data/femnist/data``, copy the two directories ``train`` and ``test`` and paste them in ``datasets/FEMNIST``   
   - # training data is varied over clients
   - # testing data = 624 
   - # classes = 62
   - # channels = 1   
   - # pixels = 28
   - # clients = 203 (in this example)
   - 

2. In the code, we set ``num_clients = 4`` as an example. Additionally, we set ``num_pixel=32``.
3. This code generates ``json`` files in the ``datasets/PreprocessedData/Coronahack_Clients_4`` directory.


How to feed a dataset into our framework APPFL?

1. Store datasets in ``appfl/datasets``
2. Create classes for reading the datasets in ``appfl/read``
3. Create a configuration file that specifies the datasets in ``appfl/config/dataset``
4. Load a testing dataset to a server as in a ``run_server`` function in ``appfl/run.py``

Code snippet:

.. code-block:: python     

    server = eval(cfg.fed.servername)(
    copy.deepcopy(model), 
    num_clients, 
    device, 
    dataloader=dataloader, 
    **cfg.fed.args,
    )

5. Load a training dataset to a client as in a ``run_client`` function in ``appfl/run.py``

Code snippet:

.. code-block:: python     

    clients = [
    eval(cfg.fed.clientname)(
    cid,
    copy.deepcopy(model),            
    optimizer,
    cfg.optim.args,
    dataloaders[i],
    device,
    **cfg.fed.args,
    )
    for i, cid in enumerate(num_client_groups[comm_rank - 1])
    ]

Important Note: ``dataloader`` and ``dataloaders[i]`` in the above code snippets are testing and training datasets constructed based on the **torch tensors**.

**Example 1. FEMNIST** 

1. Store the FEMNIST dataset in ``appfl/datasets`` (more details are in **README.md** in ``appfl/datasets``)  
   
   - The FEMNIST dataset is distributed over multiple clients
   - For every client i, #training data points = N\ :sub:`i`, #testing data points = M\ :sub:`i`
   - #clients = 203 (e.g., a small size FEMNIST)   
   - #classes of image data = 62
   - Each image data has C=1 channel, W=28 pixels width, H=28 pixels height     

2. Create classes for reading the datasets in ``appfl/read`` (see **femnist.py** in ``appfl/read``)

   - We construct two dictionaries ``train_data_image`` and ``train_data_class`` which take a client as an input and provide a set of image data and their corresponding labels, respectively. 
   - ``dataloader`` is a set of tuples, each of which is composed of ``train_data_image[client]`` and ``train_data_class[client]`` for every client.

3. Create a configuration file that specifies the datasets in ``appfl/config/dataset`` (see **femnist.yaml** in ``appfl/config/dataset``)

4. Load a testing dataset to a server as in a ``run_server`` function in ``appfl/run.py``
   
5. Load a training dataset to a client as in a ``run_client`` function in ``appfl/run.py``

- Before loading, check if
    - the dataset in ``dataloader`` is based on **torch tensors** 
    - the image data has a [N\ :sub:`i`, C, W, H] shape and the data label has [N\ :sub:`i`]

Code:

.. code-block:: python  

    if comm_rank == 1:
        for i, cid in enumerate(num_client_groups[comm_rank - 1]):
            for img, class_id in dataloaders[i]:
                print("client=", cid, "image=", img.shape, " class_id=", class_id.shape)

Output:

>>> client= 0 image= torch.Size([161, 1, 28, 28])  class_id= torch.Size([161])
client= 1 image= torch.Size([70, 1, 28, 28])  class_id= torch.Size([70])
client= 2 image= torch.Size([164, 1, 28, 28])  class_id= torch.Size([164])
client= 3 image= torch.Size([150, 1, 28, 28])  class_id= torch.Size([150])
client= 4 image= torch.Size([142, 1, 28, 28])  class_id= torch.Size([142])
client= 5 image= torch.Size([68, 1, 28, 28])  class_id= torch.Size([68])


**Example 2. Coronahack-Chest-XRay** 

1. Store the Coronahack dataset in ``appfl/datasets`` (more details are in **README.md** in ``appfl/datasets``)     
   
   - The Coronahack dataset is centralized, hence the dataset should be distributed over multiple clients.
   - #training data points = 5286, #testing data points = 624
   - #clients = P (e.g., by setting P=4, #training data points will be distributed over 4 clients)
   - #classes of image data = 7 
   - Each image data has C=3 channels, W pixels width, H pixels height (e.g., W=H=32 can be chosen)

2. Create classes for reading the datasets in ``appfl/read`` (see **coronahack.py** in ``appfl/read``)

   - We convert the raw image data such that the ``DataLoader`` from PyTorch can be utilized
   - As an example, see https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d
   
3. Create a configuration file that specifies the datasets in ``appfl/config/dataset`` (see **coronahack.yaml** in ``appfl/config/dataset``)

4. Load a testing dataset to a server as in a ``run_server`` function in ``appfl/run.py``
   
5. Load a training dataset to a client as in a ``run_client`` function in ``appfl/run.py``
 