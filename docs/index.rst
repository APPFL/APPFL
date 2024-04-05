.. APPFL documentation master file, created by
   sphinx-quickstart on Fri Oct 29 15:16:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
APPFL
=====

APPFL, Advanced Privacy-Preserving Federated Learning, is an open-source software framework that allows research communities to implement, test, and validate various ideas related to privacy-preserving federated learning (FL), and deploy real FL experiments safely and easily among distributed clients to train ML models.
With this framework, developers and/or users can easily

* Train any user-defined machine learning model on decentralized data with optional differential privacy and client authentication,
* Simulate various synchronous, asynchronous, and semi-asynchronous PPFL algorihtms on high-performance computing (HPC) architecture with MPI, 
* Implement customizations in a plug-and-play manner for all aspects of FL, including aggregation algorithms, server scheduling strategies, and client local trainers.

Technical Components
====================

APPFL is primarily composed of the following six technical components: *Aggregator*, *Scheduler*, *Trainer*, *Privacy*, *Communicator*, and *Compressor*, all with easy interface for user customizations.

.. grid:: 3

   .. grid-item-card:: 
      
      Aggregator
      ^^^^^^^^^^
      Supports several popular algorithms to aggregate one or several client local models.

   .. grid-item-card:: 
      
      Scheduler
      ^^^^^^^^^
      Supports several synchronous, asynchronous, and semi-asynchronous scheduling algorithms at the server-side to deal with different arrival times of client local models. 

   .. grid-item-card:: 
      
      Trainer
      ^^^^^^^
      Supports several client local trainers for various training tasks.

.. grid:: 3

   .. grid-item-card:: 
      
      Privacy
      ^^^^^^^
      Supports several global/local differential privacy schemes.

   .. grid-item-card:: 
      
      Communicator
      ^^^^^^^^^^^^
      Supports MPI for single-machine/cluster simulation, and gRPC and Globus Compute with authenticator for secure distributed training.

   .. grid-item-card:: 
      
      Compression
      ^^^^^^^^^^^
      Supports several lossy compressors, including `SZ2 <https://github.com/szcompressor/SZ>`_, `SZ3 <https://github.com/szcompressor/SZ3>`_, `ZFP <https://pypi.org/project/zfpy/>`_, and `SZx <https://github.com/szcompressor/SZx>`_.

APPFL aims to maintain a composable design of the package, where each technical component is extendible and independent from each other. Users can use the configuration files to specify the desired item from each component to launch FL experiments according their needs.

APPFL Framework Overview
========================

.. image:: _static/design-logic.jpg
   :width: 90%
   :align: center

In the design of the APPFL framework, we essentially create the *server agent* and *client agent*, using the six technical components above as building blocks, to act on behalf of the FL server and clients to conduct FL experiments. Specifically, 

* APPFL Server Agent: 

  * Server Config: The server configuration is provided as a single yaml file which contains not only configuration for the FL server, such as aggregator and scheduler configurations, but also configurations that apply for all clients, such as local trainer type and configurations and model architectures, which will be broadcasted to all clients at the beginning of FL experiments.
  * Compressor: If the experiment enables compression, the compressor component is used to decompress the model sent from the clients before global aggregation. 
  * Scheduler: The scheduler component handles the different arrival times of client local models, which controls the synchronism of the experiments. For example, the synchronous scheduler only passes client local models to the aggregator after all models arrive, while the asynchronous scheduler passes each client model to the aggregator immediately after it arrives.
  * Aggregator: The aggregator component takes one or more client local models, depending on the synchronism of the experiment, to update the global model.
  * Privacy: The privacy component at the server side works with the privacy component at the client side together to  provide an additional layer of protection againist data leakage.
  * Other task hanlders: In addition to the global aggregation, the FL server may also need to do other orchestration tasks, e.g., providing client-related configurations to clients. User can also define their own task handlers for their customization choices.

* APPFL Clinet Agent: 

   * Client Config: The client configuration is also provided as a yaml file for each client which contains client-specific configurations (e.g. dataloader for data on-premise), as well as uniform client configurations received from the server.
   * Compressor: If the experiment enables compression, the compressor component is used to compress the model parameters before sending to the server.
   * Trainer: The trainer component trains the machine learning model using each client's local data according to the training configurations in the client configuration.
   * Privacy: The privacy component provide an additional layer of protection againist data leakage.

* APPFL Communicator: The communicator is used for exchanging model parameters as well as metadata between the server and clients. We currently support the following three communication protocols for different use cases.
    
  * MPI: Used for simulating FL experiment on one machine or HPC cluster.
  * gRPC: Used for both simulating FL experiments on one machine or HPC, and running FL experiments on distributed machines. It can be integrated with several authentication strategies to authorize the clients for secure FL experiments.
  * `Globus Compute <https://funcx.readthedocs.io/en/latest/index.html>`_: Used for easily running FL experiments on distributed and heterogeneous machines. It is integrated with Globus authentication to authorize the clients for secure FL experiments.


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
