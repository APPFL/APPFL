# Examples

This directory contains examples showing how to run our APPFL using various communication protocols, such as MPI, gRPC, and GlobusCompute.
Currently, we have the following examples:

- MNIST
- CIFAR10
- Coronahack
- FEMNIST
- CELEBA
- Federated load forecasting with personalization layers
- FLamby Datasets

## Prepare the Datasets
- MNIST and CIFAR10: As these two datasets are part of `torchvision`, so you only need to install the package `pip install torchvision`, and then those datasets will be automatically downloaded for you when you first run the experiments scripts on those datasets.
- Coronahack is a dataset from Kaggle, you can follow the download instructions [here](datasets/RawData/README.md) to manually download it before running the corresponding experiment scripts.
- FEMNIST and CELEBA are two datasets from FL benchmark datasets [LEAF](https://github.com/TalwalkarLab/leaf/tree/master), you can follow the download instructions [here](datasets/RawData/README.md) to manually download them before running the corresponding experiment scripts.
- Federated load forecasting with personalization layers uses power consumption data from NREL datasets, which will be automatically downloaded for you when you first run the experiments script. 
- FLamby is a benchmarking dataset from cross-silo federated learning, and you need to download the datasets manually by following the [FLamby instructions](https://github.com/owkin/FLamby).


## How to run

We use separate python files for examples on various datasets with different communication protocols. For example, 

- `mnist_serial.py` contains example on the partitioned MNIST dataset using serial simulation for synchronous federated learning.
- `mnist_mpi_sync.py` contains example on the partitioned MNIST dataset using MPI for synchronous federated learning.
- `mnist_mpi_async.py` contains example on the partitioned MNIST dataset using MPUI for asynchronous federated learning.
- `mnist_mpi_privacy.py` contains example on the partitioned MNIST dataset using MPI for synchrnous privacy-preserving federated learning.
- `mnist_grpc.py` contains example on the partitioned MNIST dataset using gRPC for synchronous federated learning.
- `mnist_globus_compute` contains example on the partitioned MNIST dataset using [Globus Compute](https://funcx.readthedocs.io/en/latest/) for synchronous federated learing.


### MNIST, CIFAR10, Coronahack, CELEBA

All the examples require the same number of MPI processes.
Below shows how to execute the MNIST example.
For Coronahack and FEMNIST check the dataset directory for preprocessing the data and replace the file name with an appropriate example name.


#### MPI Communication

```bash
mpiexec -np 6 python ./mnist_mpi_sync.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10
```

#### gRPC Communication

For gRPC communication, we launch multiple MPI processes as well for simulation purposes only.
Note that our gPRC implementation itself does not require any MPI communication.

```bash
mpiexec -np 6 python ./mnist_grpc.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10
```

#### Globus Compute Communication
Globus Compute is used to support federated learning among real-world distributed and heterogeneous computing facilities. Please see the detialed instructions about how to setup experiments using Globus Compute [here](globus_compute/READMD.md).
```bash
python mnist_gc.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml
```

#### Running Serial

```bash
python ./mnist_serial.py --num_clients 5 --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10
```

### FEMNIST

#### MPI communication

```bash
mpiexec -n 204 python femnist_mpi.py --server ServerFedAvg --num_epochs 6 --client_lr 0.01
```
#### gRPC communication

```bash
mpiexec -n 204 python femnist_grpc.py
```

### Federated load forecasting with Personalization Layers

Personalization layers allow certain layers of the model to remain local to each client. For example, the load forecasting model is based on a LSTM+fully connected architecture, with the following layer names: 

```lstm_model.weight_ih_l0,lstm_model.weight_hh_l0,lstm_model.bias_ih_l0,lstm_model.bias_hh_l0,lstm_model.weight_ih_l1,lstm_model.weight_hh_l1,lstm_model.bias_ih_l1,lstm_model.bias_hh_l1,FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,prelu1.weight,prelu2.weight```

Suppose we want to personalize the fully connected layer's weights in a MPI setting, then this can be achieved as follows.

#### MPI communication

```
mpiexec -n 43 python personalization_fedloadforecast.py --personalization_layers FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,prelu1.weight,prelu2.weight --personalization_config_name MyPersonalization
```

#### Running Serial

```
python personalization_fedloadforecast.py --personalization_layers FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,prelu1.weight,prelu2.weight --personalization_config_name MyPersonalization
```

Currently, gRPC has not been implemented for personalization layers.