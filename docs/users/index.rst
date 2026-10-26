User Guide
==========

Framework Overview
------------------

.. image:: ../_static/design-logic-new.jpg
   :width: 90%
   :align: center

The APPFL framework is built around a *server agent* and *client agents*, which act on behalf of the FL server and clients to conduct FL experiments. Specifically,

* APPFL Server Agent:

  * Server Config: The server configuration is provided as a single YAML file. It contains not only configurations for the FL server itself, such as the aggregator and scheduler, but also configurations shared by all clients, such as the local trainer type, training settings, and model architecture, which are broadcast to all clients at the beginning of an FL experiment.
  * Compressor: If compression is enabled, the compressor component decompresses the models sent by the clients before global aggregation.
  * Scheduler: The scheduler component handles the different arrival times of client local models and thereby controls the synchronicity of the experiment. For example, the synchronous scheduler passes client local models to the aggregator only after all models have arrived, while the asynchronous scheduler passes each client model to the aggregator as soon as it arrives.
  * Aggregator: The aggregator component takes one or more client local models, depending on the synchronicity of the experiment, to update the global model.
  * Privacy: The server-side privacy component works together with the client-side privacy component to provide an additional layer of protection against data leakage.
  * Other task handlers: In addition to global aggregation, the FL server may need to perform other orchestration tasks, e.g., providing client-related configurations to clients. Users can also define their own task handlers for customized tasks.

* APPFL Client Agent:

   * Client Config: Each client has its own YAML configuration file, which contains client-specific configurations (e.g., the dataloader for on-premises data). It is merged with the shared client configurations received from the server.
   * Compressor: If compression is enabled, the compressor component compresses the model parameters before they are sent to the server.
   * Trainer: The trainer component trains the machine learning model on the client's local data according to the training configurations.
   * Privacy: The privacy component provides an additional layer of protection against data leakage.

* APPFL Communicator: The communicator exchanges model parameters and metadata between the server and clients. We currently support the following three communication protocols for different use cases.

  * MPI: Used for simulating FL experiments on a single machine or an HPC cluster.
  * gRPC: Used both for simulating FL experiments on a single machine or HPC cluster and for running FL experiments on distributed machines. It can be integrated with several authentication strategies to authorize the clients for secure FL experiments.
  * `Globus Compute <https://funcx.readthedocs.io/en/latest/index.html>`_: Used for easily running FL experiments on distributed and heterogeneous machines. It is integrated with Globus authentication to authorize the clients for secure FL experiments.

APPFL Components
----------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   technical_components

User and Developer Guide
------------------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   user_run
   ../developer/index

[DEPRECATED] Guide for APPFL v0
-------------------------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   deprecated/index
