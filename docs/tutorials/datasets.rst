Preprocessing datasets
======================

Our APPFL framework reads datasets, each of which is stored in a ``json`` file with the following dictionary format:

.. code-block:: console
      
   {  
   "x": [ [data input 1], [data input 2], ..., [data input N] ], 
   "y": [ data label 1, data label 2, ..., data label N ]  
   }

Therefore, the users of our APPFL framework should *preprocess their raw dataset* to create the ``json`` files with the above dictionary format. After preprocessing, the users will 
create  

- ``all_test_data.json`` (a testing dataset for **a server**) and 
- ``all_train_data_client_${i}.json`` (a set of training datasets where "i" represents **a client "i"**), 

and store the ``json`` files in a ``datasets/PreprocessedData/[DatasetName]_Clients_[#Clients]`` directory, where "DatasetName" and "#Clients" are determined by the users. 


.. In a federated learning, a server utilizes a **testing dataset** to evaluate a global model parameter updated *iteratively* based on local model parameters trained by multiple clients using their own **training dataset**.


Examples of preprocessing
-------------------------
 
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

**Example 2.** ``datasets/CIFAR10_Preprocess.py``

1. This code downloads CIFAR10 datasets from TorchVision and store them in ``datasets/RawData``. 
   
   - # training data = 50000
   - # testing data = 10000 
   - # classes = 10
   - # channels = 3 
   - # pixels = 32

2. As an example, we set ``num_clients = 4``. 

3. This code generates ``json`` files in the ``datasets/PreprocessedData/CIFAR10_Clients_4`` directory.

**Example 3.** ``datasets/Coronahack_Preprocess.py``

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

**Example 4.** ``datasets/FEMNIST_Preprocess.py``

1. Prerequisite:
 
   - ``mkdir FEMNIST`` in ``datasets/RawData``
   - Git clone https://github.com/TalwalkarLab/leaf.git
   - As an example, in ``leaf/data/femnist``, do ``./preprocess.sh -s niid --sf 0.05 -k 0 -t sample`` which downloads **a small-sized dataset**
   - In a newly generated directory ``leaf/data/femnist/data``, copy the two directories ``train`` and ``test`` and paste them in ``datasets/FEMNIST``   
   - # training data is varied over clients
   - # testing data = 4176 
   - # classes = 62
   - # channels = 1      

2. In this example, ``num_clients=203`` and ``num_pixel=28`` are fixed.
3. This code generates ``json`` files in the ``datasets/PreprocessedData/FEMNIST_Clients_203`` directory.
 