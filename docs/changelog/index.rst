Changelog
=========

appfl v1.0.0
------------

Version 1.0.0 of appfl was a major release that **refactored** the entire codebase to make it more modular, extensible, and functional, while remains backward compatible with the previous version. The release also included the following changes:

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

- The previous version of appfl is still supported but deprecated and no longer maintained. Users are encouraged to upgrade to the new version for better performance, functionality, and extensibility.
- Examples and tutorials for the previous version are still available in the ``examples/examples_v0`` directory of the appfl repository.