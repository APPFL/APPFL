# Examples

This directory contains examples showing how to run our APPFL using either MPI or gRPC.
Currently, we have the following examples:

- MNIST
- CIFAR10
- FEMNIST
- Coronahack
- CELEBA
- Federated load forecasting with Personalization layers

# How to run

Each example is implemented in a separate python file, e.g., `mnist.py` and `grpc_mnist.py` for the MNIST example.
By default, the number of clients are set to `4` for all examples, except for the FEMNIST example which is set to `203`.

We simulate federated learning by launching MPI processes (# processes = 1 + # clients, where 1 for server).
MPI protocol will be used for communication between server and clients.
For files starting with `grpc_`, we use gRPC protocol for communication instead.

## MNIST, CIFAR10, Coronahack, CELEBA

All the examples require the same number of MPI processes.
Below shows how to execute the MNIST example.
For Coronahack and FEMNIST check the dataset directory for preprocessing the data and replace the file name with an appropriate example name.


### MPI communication

```bash
mpiexec -n 5 python mnist.py --server ServerFedAvg --num_epochs 6 --client_lr 0.01
```
`--server : ServerFedAvg  or  ICEADMMServer  or IIADMMServer`
start tensorborad and then go to the web page
```shell
tensorboard --logdir=runs
```

### gRPC communication

For gRPC communication, we launch multiple MPI processes as well for simulation purposes only.
Note that our gPRC implementation itself does not require any MPI communication.

```bash
mpiexec -n 5 python grpc_mnist.py
```

### Running Serial

```bash
python mnist_no_mpi.py
```

## FEMNIST

### MPI communication

```bash
mpiexec -n 204 python femnist.py --server ServerFedAvg --num_epochs 6 --client_lr 0.01
```
`--server : ServerFedAvg  or  ICEADMMServer  or IIADMMServer`
### gRPC communication

```bash
mpiexec -n 204 python grpc_femnist.py
```

## Federated load forecasting with Personalization Layers

Personalization layers allow certain layers of the model to remain local to each client. For example, the load forecasting model is based on a LSTM+fully connected architecture, with the following layer names: 

```lstm_model.weight_ih_l0,lstm_model.weight_hh_l0,lstm_model.bias_ih_l0,lstm_model.bias_hh_l0,``` ```lstm_model.weight_ih_l1,lstm_model.weight_hh_l1,lstm_model.bias_ih_l1,lstm_model.bias_hh_l1,``` ```FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,``` ```prelu1.weight,prelu2.weight```

Suppose we want to personalize the fully connected layer's weights in a MPI setting, then this can be achieved as follows.

### MPI communication

```mpiexec -n 43 python personalization_fedloadforecasting.py --personalization_layers FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,prelu1.weight,prelu2.weight --personalization_config_name MyPersonalization```

### Running Serial

```python personalization_fedloadforecasting.py --personalization_layers FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,prelu1.weight,prelu2.weight --personalization_config_name MyPersonalization```

Currently, gRPC has not been implemented for personalization layers.