Algorithm Implementation Playbook
==================================

This guide walks through implementing a new FL algorithm in ``appfl.sim`` and exporting it as a production-ready APPFL plugin.

Naming Convention
-----------------

Algorithm components are resolved **strictly by name** from ``algorithm.name`` in config:

.. code-block:: text

   algorithm.name = fedavg  →  FedavgAggregator, FedavgScheduler, FedavgTrainer
   algorithm.name = swts    →  SwtsAggregator,   SwtsScheduler,   SwtsTrainer
   algorithm.name = swucb   →  SwucbAggregator,  SwucbScheduler,  SwucbTrainer

The PascalCase of the algorithm name is the prefix for all three component classes.
New algorithms must provide all three classes — even if the scheduler or trainer simply inherits from the base defaults.

Step 0: Sanity Check
--------------------

Before writing any code, confirm the environment is working:

.. code-block:: bash

   python -m appfl.sim.runner \
     --config src/appfl/sim/config/examples/split/mnist_iid.yaml

If this runs cleanly, you are ready to develop.

Step 1: What to Implement
--------------------------

Most new FL algorithms need three pieces:

**Aggregator (required)**

.. code-block:: text

   File:    src/appfl/algorithm/aggregator/<algo>_aggregator.py
   Inherit: BaseAggregator  (from appfl.algorithm.aggregator)
   Methods: aggregate(...), get_parameters(...)

**Trainer (optional — only if your local update is custom)**

.. code-block:: text

   File:    src/appfl/algorithm/trainer/<algo>_trainer.py
   Inherit: BaseTrainer  (from appfl.algorithm.trainer)
   Methods: train(...), get_parameters(...)

**Scheduler (optional — only for async or custom scheduling)**

.. code-block:: text

   File:    src/appfl/sim/algorithm/scheduler/<algo>_scheduler.py
   Inherit: BaseScheduler  (from appfl.sim.algorithm.scheduler)
   Methods: schedule(...)

.. note::

   You do **not** need to modify ``runner.py`` for new algorithms.

Step 2: Minimal Implementation
-------------------------------

**Create your aggregator**

Copy the FedAvg aggregator as a starting point:

.. code-block:: bash

   cp src/appfl/algorithm/aggregator/fedavg_aggregator.py \
      src/appfl/algorithm/aggregator/myalgo_aggregator.py

Rename the class to ``MyalgoAggregator`` and implement your aggregation logic.

**Register the class**

Add the new class to ``src/appfl/algorithm/aggregator/__init__.py`` (and similarly for trainer/scheduler if custom).

**Add a config file**

Create ``src/appfl/sim/config/algorithms/<algo>.yaml``:

.. code-block:: yaml

   algorithm:
     name: myalgo
   train:
     num_clients: 3
     num_rounds: 2
     num_sampled_clients: 3

**Run a smoke test** (start with serial backend, 2–3 clients, 1–2 rounds):

.. code-block:: bash

   python -m appfl.sim.runner \
     --config src/appfl/sim/config/algorithms/myalgo.yaml \
     experiment.backend=serial \
     train.num_rounds=2 train.num_clients=3 train.num_sampled_clients=3

Step 3: Custom Dataset (Optional)
----------------------------------

To use your own dataset, set ``dataset.backend=custom`` and point to your loader function:

.. code-block:: yaml

   dataset:
     backend: custom
     configs:
       custom_dataset_loader: mypackage.mymodule:load_fn

Your loader function must return:

.. code-block:: python

   client_datasets, server_dataset, dataset_meta

Where:

* ``client_datasets``: list of ``(train_dataset, test_dataset)`` or ``(train_dataset, val_dataset, test_dataset)`` per client.
* ``server_dataset``: global evaluation dataset, or ``None``.
* ``dataset_meta``: namespace with at least ``num_clients``, ``num_classes``, ``input_shape``.

.. tip::

   For image or audio data, ``dataset_meta`` should have ``need_embedding=False``.
   For text or token data, set ``need_embedding=True`` and also provide ``seq_len`` and ``num_embeddings`` so that embedding-based models are configured correctly.

Step 4: Custom Metrics (Optional)
----------------------------------

**Easy path** — just return ``loss``, ``accuracy``, and ``num_examples`` from your train/eval loops.

**Registry path** — add a class to ``src/appfl/metrics/metricszoo.py``:

.. code-block:: python

   class MyMetric(BaseMetric):
       def collect(self, pred, true):
           ...
       def summarize(self):
           ...

Then reference it in config: ``eval.metrics: [mymetric]``.

Step 5: Common Mistakes
------------------------

Before deep debugging, check these first:

* Class name matches ``algorithm.name`` via PascalCase exactly (e.g., ``fedavg`` → ``FedavgAggregator``, not ``FedAvgAggregator``).
* Dataset loader returns ``(train_dataset, test_dataset)`` per client, not the raw dataset.
* ``num_classes`` and ``input_shape`` in ``dataset_meta`` are correct and match the model's expected input.
* ``model.name`` is an exact class name (case-sensitive); fuzzy aliases are not resolved.
* Start with ``backend=serial`` before switching to ``nccl``/``gloo``.

Step 6: Export to APPFL (Production)
--------------------------------------

Once your algorithm is validated in simulation, export it as a plug-and-play APPFL plugin using the exporter script:

.. code-block:: bash

   # Safe mode: generate artifact folder without modifying APPFL source
   python -m appfl.sim.tools.export_appfl_plugin \
     --algorithm myalgo \
     --aggregator-source src/appfl/algorithm/aggregator/myalgo_aggregator.py \
     --aggregator-class MyalgoAggregator \
     --trainer-source src/appfl/algorithm/trainer/myalgo_trainer.py \
     --trainer-class MyalgoTrainer \
     --scheduler-source src/appfl/sim/algorithm/scheduler/myalgo_scheduler.py \
     --scheduler-class MyalgoScheduler \
     --output-dir build/appfl_plugin_myalgo

   # Direct mode: write directly into the APPFL source tree
   python -m appfl.sim.tools.export_appfl_plugin \
     --algorithm myalgo \
     --aggregator-source src/appfl/algorithm/aggregator/myalgo_aggregator.py \
     --aggregator-class MyalgoAggregator \
     --appfl-root .

The exporter produces:

* APPFL-style module files under ``src/appfl/algorithm/...``
* Auto-vendored metric classes when the trainer depends on ``MetricsManager``
* A config template under ``src/appfl/sim/config/algorithms/<algo>.yaml``
* Patch/install instructions (artifact mode)

.. tip::

   Run with ``--check-appfl-root /path/to/APPFL`` to get a compatibility audit before writing any files.

Algorithm Backends at a Glance
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Backend
     - Transport
     - Recommended use
   * - ``serial``
     - Single process
     - All initial development and debugging. Deterministic, no CUDA requirements.
   * - ``nccl``
     - torch.distributed + NCCL
     - Multi-GPU scale-up. Use ``CUDA_VISIBLE_DEVICES`` to select GPU subset.
   * - ``gloo``
     - torch.distributed + Gloo
     - CPU multi-process. Useful for testing distributed logic without GPUs.
