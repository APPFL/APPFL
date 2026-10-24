Basic PPFL Training
===================

``APPFL`` enables users to simulate and train PPFL on a single machine, a cluster, or multiple heterogeneous machines. We use the terms

- **simulation** for running PPFL experiments on a single machine or a cluster without actual data decentralization
- **training** for running PPFL experiments on multiple (heterogeneous) machines with actual decentralization of client datasets

We provide examples of both simulation and training in the following sections. In particular, simulation is useful for those who develop, test, and validate new models and algorithms for PPFL, whereas training is for those who deploy PPFL in real-world settings.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_simulation_serial
   user_simulation_mpi
   user_train

In addition to the examples above, we also provide instructions on how to define models, loss functions, metrics, and datasets for your own FL applications.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_model
   user_loss
   user_metric
   user_data
