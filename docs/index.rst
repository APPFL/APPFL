.. APPFL documentation master file, created by
   sphinx-quickstart on Fri Oct 29 15:16:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
APPFL
=====

APPFL, Advanced Privacy-Preserving Federated Learning, is an open-source software framework that allows research communities to implement, test, and validate various ideas for privacy-preserving federated learning (FL). 
With this framework, developers and/or users can 

* Train a user-defined neural network model on decentralized data with differential privacy, 
* Simulate various PPFL algorithms on high-performance computing (HPC) architecture with MPI, 
* Implement user-defined PPFL algorithms in a plug-and-play manner. 

Technical Components
====================

The core technical components of APPFL include algorithms, privacy-preserving techniques, communication protocols, compression techniques, FL model to train, and data.

.. grid:: 4

   .. grid-item-card:: 
      
      Algorithm
      ^^^^^^^^^
      Supports several synchronous/asynchronous algorihtms with privacy schemes

   .. grid-item-card:: 
      
      Privacy
      ^^^^^^^
      Supports several global/local differential privacy schemes

   .. grid-item-card:: 
      
      Communication
      ^^^^^^^^^^^^^
      Supports MPI, gRPC, and Globus Compute

   .. grid-item-card:: 
      
      Compression
      ^^^^^^^^^^^
      Supports several lossy compressors, including SZ2, SZ3, SZx, and ZFP

APPFL aims to maintain a composable design of the package, where each technical component is indepent each other. Any combination of the components can ideally run without any modiciation of the package. For example, an FL experiment that run with FedAvg with SZ2 compression can run with FedAsync with SZx. The package's composability provides a number of combinations of algorithmic choices for PPFL experiments.

APPFL Framework Overview
========================

.. image:: _static/design.jpg
   :width: 50%
   :align: center

In the design of APPFL framework, we decompose an execution of federated learning experiment into three main components, *APPFL Server*, *APPFL Communicator*, and *APPFL Client*. The details and sub-components of these three are detailed as follows:

* APPFL Server: The server orchestrates the whole FL experiment run by providing the model architecture, loss and metric, and configurations used in the training, and aggregating the client trained models synchronously or asynchronously using certain federated learning algorithms.

* APPFL Communicator: The communicator is used for exchanging metadata as well as the model weights between the server and clients. We currently support the following three communication protocols for different use cases.
    
  * MPI - Used for simulating FL experiment on one machine or HPC. MPI communicator now also supports model compression for efficient communication.
  * gRPC - Used for both simulating FL experiments on one machine or HPC, and running FL experiments on distributed machines.
  * `Globus Compute <https://funcx.readthedocs.io/en/latest/index.html>`_ - Used for easily running FL experiments on distributed and heterogeneous machines.
  * Compression - Several lossy compressors are available: `SZ2 <https://github.com/szcompressor/SZ>`_, `SZ3 <https://github.com/szcompressor/SZ3>`_, `ZFP <https://pypi.org/project/zfpy/>`_, and `SZx <https://github.com/szcompressor/SZx>`_. Please refer to their official project/GitHub pages if you want more detailed information of them. Here, we only provide the installation instructions. Note: SZx need particular permission to access because collaboration with a third-party, so we omit its installation here.

* APPFL Client: APPFL clients have local dataset on disk and and use dataloader to load them for the local trainer to train local models.


Main Topics
===========

.. toctree::
   :maxdepth: 1

   install/index
   tutorials/index
   users/index
   developer/index
   community/index


Acknowledgement
===============
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
