Changelog
=========

appfl v1.0.2
------------

New Features
~~~~~~~~~~~~

- Add a new command line interface (CLI), `appfl-setup-cli` to create necessary certificates for creating SSL-secured gRPC connections between the server and clients.
- Add a tutorial on how to use the CLI, `appfl-setup-cli`, to create certificates for the server and clients, and enable SSL-secured gRPC connections between the server and clients.
- Add a detailed step-by-step tutorial on how to define custom action with an example to generate a data readiness report on all client data at `here <https://appfl.ai/en/latest/tutorials/examples_custom_action.html>`_.
- Add a APPFL `YouTube channel <https://www.youtube.com/channel/UCzwiJboiJW3dLI0UndnDy5g>`_ to provide video tutorials on how to use APPFL for federated learning research in the future.

Bug Fixes
~~~~~~~~~

- Fix the `issue <https://github.com/APPFL/APPFL/issues/197>`_ regarding client gradient clipping. The clipping is now applied before weights update.

appfl v1.0.1
------------

New Features
~~~~~~~~~~~~

- For the aggregators, the model architecture is set to be an optional initialization parameter, and the aggregators only aggregate the parameters sent by the clients instead of the whole set of model parameters. This is useful when doing federated fine-tuning or federated transfer learning where only part of model parameters are updated / the model architecture is unknown to the aggregator.
- Support easy integration of custom trainer/aggregator: user only needs to provide the custom trainer/aggregator class name and the path to the definition file in the configuration file to use it, instead of modifying the source code.
- Add a detailed step-by-step tutorial on how to use ``APPFL`` to fine-tune a ViT model with a custom trainer.

appfl v1.0.0
------------

Version 1.0.0 of appfl is a major release that refactors the entire codebase to make it more modular, extensible, and functional, while remains backward compatibility with the previous version. The release also included the following changes:

New Features
~~~~~~~~~~~~

- Define server and client agents to act on behalf of the FL server and clients to conduct FL experiments.
- Simplify the configuration process for launching FL experiments by only providing a single YAML file for the server and a YAML file for each client.
- Rebuild the communicator module, supporting MPI, gRPC, and Globus Compute, to robustly exchange model parameters as well as task metadata between the server and clients in both synchronous and asynchronous FL experiment settings.
- Implement Globus-based authentication for secure distributed training with gRPC and Globus Compute - only members within the same specific Globus group can participate in the FL experiment.
- Integrate several lossy and error-bounded lossless compressors to the communicator module for efficient model compression.
- Add documentation for the new version available at `appfl.ai <https://appfl.ai>`_

Deprecated
~~~~~~~~~~

- The previous version of appfl is still seamlessly supported but deprecated and no longer maintained. Users are encouraged to upgrade to the new version for better performance, functionality, and extensibility.
- Examples and tutorials for the previous version are still available in the ``examples/examples_legacy`` directory of the Github appfl repository.