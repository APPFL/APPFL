Changelog
=========

appfl v1.1.0
------------

New Features
~~~~~~~~~~~~

- Support batched MPI, with documentation available `here <https://appfl.ai/en/latest/tutorials/examples_batched_mpi.html>`_.
- Add more data readiness metrics such as PCA plot in this `pull request <https://github.com/APPFL/APPFL/pull/208>`_.
- Backend support for `service.appfl.ai <https://appflx.link/>`_.
- Add documentation for service.appfl.ai at `here <https://appfl.ai/en/latest/tutorials/appflx/index.html>`_.
- Add logging capabilities to the server side to log the training metadata such as the training and validation losses.
- Change documentation theme to ``furo``.

appfl v1.0.5
------------

New Features
~~~~~~~~~~~~

- Add the feature to generate data readiness reports on all client data.
- Update the documentation for adding custom action at `here <https://appfl.ai/en/latest/tutorials/examples_custom_action.html>`_.

appfl v1.0.4
------------

New Features
~~~~~~~~~~~~

- Add documentation for using APPFL with Globus Compute for secure distributed training at `here <https://appfl.ai/en/latest/tutorials/examples_globus_compute.html>`_.

Bug Fixes
~~~~~~~~~

- Fix an issue with Globus Compute at this `commit <https://github.com/APPFL/APPFL/commit/705b5af64389c77e1c0f9f21d1d86c0cc33cd067>`_.

appfl v1.0.3
------------

New Features
~~~~~~~~~~~~

- Add trackback information to the gRPC server to help debug the server-side errors.
- Add a video tutorials for `installing APPFL on AWS <https://youtu.be/ihPofoQwUMs>`_, `creating SSL-encrypted gRPC server <https://youtu.be/3n8a026VqdQ>`_, and `using APPFL to finetune a ViT <https://youtu.be/m4rdOub2Y_o>`_.

Bug Fixes
~~~~~~~~~

- Handle corner cases for server aggregators when the keys in client local models are not consistent with the global model keys.

appfl v1.0.2
------------

New Features
~~~~~~~~~~~~

- Add a new command line interface (CLI), `appfl-setup-ssl` to create necessary certificates for creating SSL-secured gRPC connections between the server and clients.
- Add a tutorial on how to use the CLI, `appfl-setup-ssl`, to create certificates for the server and clients, and enable SSL-secured gRPC connections between the server and clients.
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
