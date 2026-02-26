Quick Start
===========

Installation
------------

Install from source with the ``[sim]`` or ``[sim-all]`` extra:

.. code-block:: bash

   git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
   cd APPFL
   pip install -e ".[sim]"       # minimal: core simulation dependencies
   pip install -e ".[sim-all]"   # full: adds HuggingFace, MedMNIST, TorchAudio, TensorBoard, and more

Two equivalent entry points are available after install:

.. code-block:: bash

   # Short CLI alias (registered by pip install)
   appfl-sim --config src/appfl/sim/config/examples/split/mnist_iid.yaml

   # Module invocation (works whenever the package is importable)
   python -m appfl.sim.runner --config src/appfl/sim/config/examples/split/mnist_iid.yaml

Both call the same ``main()`` function and accept identical arguments.
``appfl-sim`` requires the package to be installed (``pip install -e``), while
``python -m appfl.sim.runner`` works as long as the repo is on the Python path.

Running an Experiment
---------------------

The entry point is ``appfl.sim.runner``, invoked as a Python module:

.. code-block:: bash

   python -m appfl.sim.runner \
     --config src/appfl/sim/config/examples/split/mnist_iid.yaml

Use CLI dot-list overrides to tweak parameters without editing YAML:

.. code-block:: bash

   python -m appfl.sim.runner \
     --config src/appfl/sim/config/examples/mnist_quickstart.yaml \
     experiment.backend=serial \
     experiment.device=cpu \
     experiment.server_device=cpu \
     train.num_rounds=1 train.num_clients=2 train.num_sampled_clients=2

Config is always resolved in three layers:

1. **Base defaults** — ``src/appfl/sim/config/examples/simulation.yaml``
2. **User config** — the ``--config`` file
3. **CLI overrides** — dot-list tokens after the config path

Multi-GPU (NCCL)
----------------

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1 python -m appfl.sim.runner \
     --config src/appfl/sim/config/algorithms/fedavg.yaml

Set ``experiment.backend=nccl`` and ``experiment.device=cuda`` in your config or as CLI overrides.

Multi-Process CPU (Gloo)
------------------------

.. code-block:: bash

   appfl-sim --config src/appfl/sim/config/examples/backend/gloo.yaml

No ``torchrun`` or ``mpirun`` is needed. The runner internally spawns CPU worker processes via ``torch.multiprocessing``, so a single ``appfl-sim`` (or ``python -m appfl.sim.runner``) invocation is sufficient. ``experiment.device=cpu`` is set automatically in the Gloo config.

Config Examples
---------------

Example configs shipped under ``src/appfl/sim/config/examples/``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Description
   * - ``examples/simulation.yaml``
     - Base template with all defaults
   * - ``examples/mnist_quickstart.yaml``
     - Minimal MNIST smoke-test config
   * - ``examples/split/mnist_iid.yaml``
     - MNIST with IID split
   * - ``examples/split/mnist_dirichlet.yaml``
     - MNIST with Dirichlet non-IID split
   * - ``examples/split/mnist_pathological.yaml``
     - MNIST with pathological non-IID split
   * - ``examples/backend/nccl.yaml``
     - Multi-GPU NCCL backend
   * - ``examples/backend/gloo.yaml``
     - CPU multi-process Gloo backend
   * - ``examples/logging/mnist_iid_wandb.yaml``
     - WandB logging example
   * - ``examples/logging/mnist_iid_tensorboard.yaml``
     - TensorBoard logging example
   * - ``examples/evaluation/mnist_holdout_dataset_eval.yaml``
     - Dataset-level federated evaluation
   * - ``examples/evaluation/mnist_holdout_client_eval.yaml``
     - Client-level holdout evaluation
   * - ``algorithms/fedavg.yaml``
     - FedAvg algorithm config
   * - ``algorithms/swts.yaml``
     - SWTS algorithm config
   * - ``algorithms/swucb.yaml``
     - SWUCB algorithm config
