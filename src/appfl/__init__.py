"""
APPFL is a privacy-preserving federated learning framework that allows users to implement a federated learning environment with 

* user-defined neural network (based on ``torch.nn.Module``) and loss function
* various federated learning algorithms
* privacy-preserving techniques
* custom decentralized data

The framework is designed to run on a single machine (laptop or cluster) for simulation as well as multiple heterogeneous machines for real deployment.
"""
from . import run_serial, run_mpi, run_mpi_async, run_mpi_sync, run_mpi_compass, run_grpc_client, run_grpc_server, run_globus_compute_server

__version__ = '1.0.0'
__author__ = 'Argonne National Laboratory'
__all__ = [
    'run_serial', 
    'run_mpi', 
    'run_mpi_async', 
    'run_mpi_sync', 
    'run_mpi_compass', 
    'run_grpc_client', 
    'run_grpc_server', 
    'run_globus_compute_server'
]