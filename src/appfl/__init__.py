"""
APPFL is a privacy-preserving federated learning framework that allows users to implement a federated learning environment with customized algorithms and privacy techniques. The framework is designed to run on a single machine (a laptop or a cluster) as well as multiple heterogeneous machines.

The framework consists of the five modules:

* :mod:`appfl.algorithm`: algorithms to train a global model (e.g., FedAvg)
* :mod:`appfl.privacy`: privacy techniques (TBD)
* :mod:`appfl.protos`: communication protocols

The configuration of the framework is set in:

* :mod:`appfl.config`: configuration for running the framework

"""

from appfl import *
