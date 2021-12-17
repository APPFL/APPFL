# APPFL

APPFL is a privacy-preserving federated learning framework that provides an infrastructure to implement algorithmic components for federated learning.

Basically, all algorithms are implemented under the `appfl` directory.
Currently, we support two algorithms for federated learning: federated averaging and inexact ADMM.
You could find more details about them in `appfl/algorithm` directory.

For communications between server and clients, we support MPI and gRPC communication protocols.
MPI is for cluster environment, and gRPC can be used when we have clients on remote or mixed platforms.
More details on our gRPC protocols can be found in the `appfl/protos` directory.

We are in progress to add privacy-preserving features, which will be available in the `appfl/privacy` directory.

The `examples` directory introduces how one can perform federated learning using our package.
We have currently tested MNIST, CIFAR10, FEMNISt, and Coronahack.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
