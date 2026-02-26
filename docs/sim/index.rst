[New] Simulation Toolkit
=======================

The **APPFL Simulation Toolkit** (``appfl.sim``) is a lightweight federated learning simulation engine built directly into APPFL.
Its design philosophy is: 

*PoC here, port later* 

Users can prototype and validate a new FL algorithm locally in hours, then automatically export it as a plug-and-play APPFL component for real distributed deployment via gRPC, MPI, or Globus Compute.

.. grid:: 3

   .. grid-item-card::

      Algorithm Skeleton
      ^^^^^^^^^^^^^^^^^^
      Standard APPFL ``aggregator`` / ``scheduler`` / ``trainer`` pattern with config-driven wiring and a strict PascalCase naming convention.

   .. grid-item-card::

      Rich Dataset Support
      ^^^^^^^^^^^^^^^^^^^^
      Built-in parsers for TorchVision, HuggingFace, LEAF, MedMNIST, Flamby, TFF, and custom loaders, with IID/non-IID federated split strategies.

   .. grid-item-card::

      Flexible Backends
      ^^^^^^^^^^^^^^^^^
      Run serially on a laptop or scale to multi-GPU nodes via ``torch.distributed`` (NCCL/Gloo) — same config, same code.

.. grid:: 3

   .. grid-item-card::

      Model Zoo
      ^^^^^^^^^
      Bundled FL benchmark models: TwoCNN, TwoNN, LeNet, LogReg, SimpleCNN, LSTM, GRU, Transformer, UNet3D, and more.

   .. grid-item-card::

      Metrics & Logging
      ^^^^^^^^^^^^^^^^^
      17 built-in metrics (``acc1``, ``auroc``, ``f1``, ``dice``, ``mse``, …). Experiment tracking via file, console, TensorBoard, or WandB.

   .. grid-item-card::

      Privacy & Security
      ^^^^^^^^^^^^^^^^^^
      Differential privacy (Laplace, Gaussian, Opacus) and pairwise-masking secure aggregation as drop-in config flags.

Architecture
------------

The simulation runtime is organized into five layers:

**Runner** (``appfl.sim.runner``)
   The main entry point. ``parse_config()`` merges the three config layers (base defaults → user YAML → CLI overrides), then dispatches to ``run_serial()`` or ``run_distributed()`` depending on ``experiment.backend``. No changes to the runner are needed when adding new algorithms.

**Agent layer** (``appfl.sim.agent``)
   ``ServerAgent`` wires the model, loss function, aggregator, scheduler, and server-side evaluation dataset. ``ClientAgent`` wraps a single ``Trainer`` instance and handles local training, evaluation, and parameter exchange with the server.

**Algorithm components** (``appfl.algorithm``, ``appfl.sim.algorithm``)
   Every FL algorithm consists of exactly three classes resolved by naming convention:

   .. list-table::
      :header-rows: 1
      :widths: 20 30 25 25

      * - Component
        - Base class
        - Required methods
        - Location
      * - Aggregator
        - ``BaseAggregator``
        - ``aggregate()``, ``get_parameters()``
        - ``appfl.algorithm.aggregator``
      * - Scheduler
        - ``BaseScheduler``
        - ``schedule()``
        - ``appfl.sim.algorithm.scheduler``
      * - Trainer
        - ``BaseTrainer``
        - ``train()``, ``get_parameters()``
        - ``appfl.algorithm.trainer``

**Loaders** (``appfl.sim.loaders``)
   ``load_dataset()`` delegates to the appropriate dataset parser backend and returns ``(client_datasets, server_dataset, dataset_meta)``. ``load_model()`` supports built-in model zoo, HuggingFace, TorchVision, and custom model files. ``dataset_meta`` carries ``num_clients``, ``num_classes``, and ``input_shape``, which feed into automatic model configuration.

**Metrics & logging** (``appfl.metrics``, ``appfl.logger``)
   ``MetricsManager`` aggregates named metric classes (``BaseMetric``) collected batch-by-batch during training and evaluation. ``ExperimentTracker`` writes structured per-round results to the configured logging backend.

Algorithm Naming Convention
---------------------------

Algorithm components are resolved **strictly by naming convention** from ``algorithm.name`` in config:

.. code-block:: text

   algorithm.name = fedavg  →  FedavgAggregator,  FedavgScheduler,  FedavgTrainer
   algorithm.name = swts    →  SwtsAggregator,    SwtsScheduler,    SwtsTrainer
   algorithm.name = swucb   →  SwucbAggregator,   SwucbScheduler,   SwucbTrainer

All three classes must be provided for every algorithm, even when the scheduler or trainer simply inherits from a base default.

Client Lifecycle Modes
----------------------

``appfl.sim`` supports two client lifecycle strategies controlled by ``experiment.stateful``:

* **Cross-device** (``stateful=false``, default): clients are constructed on-demand each round from config and released after their update is collected. Supports an optional worker-pool for reuse across rounds.
* **Cross-silo** (``stateful=true``): all clients are built once at startup and reused with preserved dataloader and optimizer state across rounds.

Execution Backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Backend
     - Transport
     - When to use
   * - ``serial``
     - Single process
     - All initial development and debugging. Deterministic, no distributed setup required.
   * - ``nccl``
     - ``torch.distributed`` + NCCL
     - Multi-GPU scale-up on a single node or cluster. Set ``experiment.device=cuda``.
   * - ``gloo``
     - ``torch.distributed`` + Gloo
     - CPU multi-process. Useful for testing distributed logic without GPUs.

In distributed mode (``nccl``/``gloo``), clients are partitioned across ranks; rank 0 runs the server and handles aggregation and logging.

.. toctree::
   :maxdepth: 1

   quickstart
   config_guide
   algorithm_playbook
