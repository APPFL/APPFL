# Examples

This directory contains examples showing how to run our APPFL using either MPI or gRPC.
Currently, we have the following examples:

- MNIST
- CIFAR10
- FEMNIST
- Coronahack

# How to run

Each example is implemented in a separate python file, e.g., `mnist.py` and `grpc_mnist.py` for the MNIST example.
By default, the number of clients are set to `4` for all examples, except for the FEMNIST example which is set to `203`.

We simulate federated learning by launching MPI processes (# processes = 1 + # clients, where 1 for server).
MPI protocol will be used for communication between server and clients.
For files starting with `grpc_`, we use gRPC protocol for communication instead.

## MNIST, CIFAR10, Coronahack

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

