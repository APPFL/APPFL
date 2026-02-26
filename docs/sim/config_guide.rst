Configuration Guide
===================

All runtime behavior of ``appfl.sim`` is controlled through YAML configuration files loaded by OmegaConf.
Config is resolved in three layers: base defaults → user ``--config`` file → CLI dot-list overrides.
The base defaults are defined in ``src/appfl/sim/config/examples/simulation.yaml``.

CLI Flag
--------

.. code-block:: bash

   python -m appfl.sim.runner --config path/to/config.yaml key=value key2=value2

``--config`` is optional. When omitted, the built-in base template is used.

Experiment (``experiment``)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``name``
     - ``appfl-sim``
     - Experiment name; used in output paths.
   * - ``seed``
     - ``42``
     - Global random seed.
   * - ``device``
     - ``cpu``
     - Device for clients (``cpu``, ``cuda``, ``cuda:0``, …).
   * - ``server_device``
     - ``cpu``
     - Device for the server.
   * - ``backend``
     - ``serial``
     - Execution backend: ``serial``, ``nccl``, or ``gloo``.
   * - ``stateful``
     - ``false``
     - ``true`` for cross-silo FL (persistent clients); ``false`` for cross-device FL (stateless clients).

Dataset (``dataset``)
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - Description
   * - ``path``
     - ``./data``
     - Dataset root path.
   * - ``name``
     - ``MNIST``
     - Dataset name.
   * - ``backend``
     - ``torchvision``
     - Loader backend: ``torchvision``, ``torchtext``, ``torchaudio``, ``medmnist``, ``leaf``, ``flamby``, ``tff``, ``hf``, ``custom``.
   * - ``download``
     - ``true``
     - Download dataset if not present.
   * - ``configs``
     - ``{}``
     - Backend-specific keyword arguments (e.g., ``raw_data_fraction``, ``min_samples_per_client``, ``terms_accepted``).

Split Simulation (``split``)
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - Description
   * - ``type``
     - ``iid``
     - Split strategy: ``iid``, ``unbalanced``, ``dirichlet``, ``pathological``, or ``pre``. Use ``pre`` when the dataset carries pre-defined client assignments (required for ``leaf``, ``flamby``, ``tff``).
   * - ``configs.dirichlet_alpha``
     - ``0.3``
     - Concentration parameter for Dirichlet split.
   * - ``configs.min_classes``
     - ``2``
     - Minimum unique classes per client for pathological split.
   * - ``configs.unbalanced_keep_min``
     - ``0.5``
     - Minimum keep ratio for unbalanced split.
   * - ``configs.pre_source``
     - ``""``
     - Column/key name that carries pre-defined client IDs (when ``split.type=pre``).
   * - ``configs.pre_index``
     - ``-1``
     - Tuple/list index used to read the client ID from each sample when samples are tuples/lists and ``split.type=pre``.
   * - ``configs.pre_infer_num_clients``
     - ``false``
     - Infer number of clients from unique IDs instead of ``train.num_clients`` (when ``split.type=pre``).

Model (``model``)
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 15 57

   * - Key
     - Default
     - Description
   * - ``name``
     - ``SimpleCNN``
     - Model name. For built-in models use class names from the model zoo (e.g., ``TwoCNN``, ``LeNet``, ``LogReg``).
   * - ``backend``
     - ``auto``
     - Model source: ``auto``, ``custom``, ``hf``, ``torchvision``, ``torchtext``, ``torchaudio``.
   * - ``path``
     - ``./models``
     - Local model root path and HuggingFace cache fallback.
   * - ``configs.num_classes``
     - inferred
     - Override class count (inferred from dataset by default).
   * - ``configs.in_channels``
     - inferred
     - Override input channel count (inferred from dataset shape by default).
   * - ``configs.hidden_size``
     - ``64``
     - Hidden dimension for applicable models.
   * - ``configs.num_layers``
     - ``2``
     - Stacked layer count for applicable models.
   * - ``configs.dropout``
     - ``0.0``
     - Dropout probability for applicable models.
   * - ``configs.seq_len``
     - ``128``
     - Sequence length for text/sequence models.
   * - ``configs.num_embeddings``
     - ``10000``
     - Vocabulary size for embedding-based models.
   * - ``configs.embedding_size``
     - ``128``
     - Embedding dimension for embedding-based models.
   * - ``configs.use_model_tokenizer``
     - ``false``
     - Use the model's own tokenizer in the text dataset parser.
   * - ``configs.cache_dir``
     - ``model.path``
     - Cache directory for external model artifacts (e.g., HuggingFace downloads).
   * - ``configs.hf_local_files_only``
     - ``false``
     - When ``true``, disallow HuggingFace network downloads and read from local cache only.

Training (``train``)
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``num_rounds``
     - ``20``
     - Number of federated rounds.
   * - ``num_clients``
     - ``20``
     - Total number of clients.
   * - ``num_sampled_clients``
     - ``4``
     - Clients sampled per round.
   * - ``update_base``
     - ``epoch``
     - Local update unit: ``epoch`` or ``iter``.
   * - ``local_epochs``
     - ``1``
     - Local epochs per client (when ``update_base=epoch``).
   * - ``local_iters``
     - ``1``
     - Local iterations per client (when ``update_base=iter``).
   * - ``batch_size``
     - ``32``
     - Local training batch size.
   * - ``shuffle``
     - ``true``
     - Shuffle the training DataLoader each round.
   * - ``eval_batch_size``
     - ``128``
     - Batch size for evaluation.
   * - ``num_workers``
     - ``0``
     - DataLoader worker processes.
   * - ``pin_memory``
     - auto
     - Base DataLoader pin-memory flag. Defaults to ``true`` on CUDA devices, ``false`` otherwise.
   * - ``train_pin_memory``
     - inherits ``pin_memory``
     - Override pin-memory for the training DataLoader only.
   * - ``eval_pin_memory``
     - inherits ``pin_memory``
     - Override pin-memory for the evaluation DataLoader only.
   * - ``dataloader_persistent_workers``
     - ``false``
     - Pass-through to ``DataLoader(persistent_workers=...)`` when ``num_workers > 0``.
   * - ``dataloader_prefetch_factor``
     - ``2``
     - Pass-through to ``DataLoader(prefetch_factor=...)`` when ``num_workers > 0``.
   * - ``max_grad_norm``
     - ``0.0``
     - Gradient clipping threshold (disabled when ``0``).

Algorithm (``algorithm``)
--------------------------

Algorithm components are resolved by a strict naming convention from ``algorithm.name``:

.. code-block:: text

   algorithm.name=fedavg  →  FedavgAggregator, FedavgScheduler, FedavgTrainer

The explicit ``aggregator``, ``scheduler``, and ``trainer`` keys can override the class name inferred by convention when needed.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - Description
   * - ``name``
     - ``fedavg``
     - Algorithm label used to infer all three component class names.
   * - ``aggregator``
     - ``""``
     - Explicit aggregator class name override (optional; inferred from ``name`` when empty).
   * - ``scheduler``
     - ``""``
     - Explicit scheduler class name override (optional; inferred from ``name`` when empty).
   * - ``trainer``
     - ``""``
     - Explicit trainer class name override (optional; inferred from ``name`` when empty).
   * - ``mix_coefs``
     - ``sample_ratio``
     - Aggregation weighting: ``uniform``, ``sample_ratio``, or ``adaptive``.
   * - ``optimize_memory``
     - ``true``
     - Enable memory-saving cleanup paths in trainer/scheduler/aggregator.
   * - ``aggregator_kwargs``
     - ``{}``
     - Forwarded to aggregator constructor.
   * - ``scheduler_kwargs``
     - ``{}``
     - Forwarded to scheduler constructor.
   * - ``trainer_kwargs``
     - ``{}``
     - Forwarded to trainer constructor.

Loss (``loss``)
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``name``
     - ``CrossEntropyLoss``
     - Loss function name (any ``torch.nn`` class, or ``custom``).
   * - ``backend``
     - ``auto``
     - Loss source: ``auto``, ``torch``, or ``custom``.
   * - ``path``
     - ``""``
     - Path for custom loss module lookup.
   * - ``configs``
     - ``{}``
     - Keyword arguments forwarded to the loss constructor.

Optimizer (``optimizer``)
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``name``
     - ``SGD``
     - Optimizer name (any ``torch.optim`` class, or ``custom``).
   * - ``backend``
     - ``auto``
     - Optimizer source: ``auto``, ``torch``, or ``custom``.
   * - ``path``
     - ``""``
     - Path for custom optimizer module lookup.
   * - ``lr``
     - ``0.01``
     - Learning rate.
   * - ``clip_grad_norm``
     - ``0.0``
     - Gradient norm clipping threshold (disabled when ``0``).
   * - ``lr_decay.enable``
     - ``false``
     - Enable a local LR scheduler applied each round.
   * - ``lr_decay.type``
     - ``none``
     - LR scheduler type: ``none``, ``exponential``, or ``cosine``.
   * - ``lr_decay.gamma``
     - ``0.99``
     - Decay factor for exponential scheduler.
   * - ``lr_decay.t_max``
     - ``0``
     - Cosine cycle length; ``≤0`` auto-sets to the local update length.
   * - ``lr_decay.eta_min``
     - ``0.0``
     - Minimum LR for cosine scheduler.
   * - ``lr_decay.min_lr``
     - ``0.0``
     - Hard floor applied to LR after every scheduler step.
   * - ``configs.weight_decay``
     - ``0.0``
     - Weight decay.
   * - ``configs.momentum``
     - ``0.0``
     - Momentum.

Evaluation (``eval``)
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``every``
     - ``1``
     - Evaluation cadence (every N rounds).
   * - ``metrics``
     - ``[acc1]``
     - Metric list. Supported: ``acc1``, ``acc5``, ``auroc``, ``auprc``, ``youdenj``, ``f1``, ``precision``, ``recall``, ``seqacc``, ``mse``, ``rmse``, ``mae``, ``mape``, ``r2``, ``d2``, ``dice``, ``balacc``.
   * - ``do_pre_evaluation``
     - ``true``
     - Evaluate the global model before local updates each round.
   * - ``do_post_evaluation``
     - ``true``
     - Evaluate the global model after aggregation each round.
   * - ``show_eval_progress``
     - ``true``
     - Show a tqdm progress bar during evaluation.
   * - ``enable_global_eval``
     - ``true``
     - Server-side evaluation on the server holdout set.
   * - ``enable_federated_eval``
     - ``true``
     - Client-side federated evaluation on per-client holdout sets.
   * - ``configs.scheme``
     - ``dataset``
     - Federated eval mode: ``dataset`` (each client's own split) or ``client`` (dedicated holdout clients).
   * - ``configs.dataset_ratio``
     - ``[80, 20]``
     - Train/val/test split ratio per client (e.g., ``[80,20]`` for train+test, ``[80,10,10]`` for train+val+test). Use ``[100]`` for train-only mode, which disables all evaluation.
   * - ``configs.client_ratio``
     - ``0.0``
     - Fraction of clients reserved as holdout when ``scheme=client``.
   * - ``configs.client_counts``
     - ``0``
     - Number of clients reserved as holdout when ``scheme=client``.

Logging (``logging``)
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - Description
   * - ``backend``
     - ``file``
     - Logging backend: ``none``, ``file``, ``console``, ``tensorboard``, ``wandb``.
   * - ``path``
     - ``./logs``
     - Output log directory. Runs are saved under ``logging.path/experiment.name/logging.name/<run_id>``.
   * - ``name``
     - ``experiment.name``
     - Run name (WandB run name when using ``wandb`` backend).
   * - ``type``
     - ``auto``
     - Logging policy. ``auto`` disables per-client logging when ``num_sampled_clients < num_clients`` (server-only for performance). ``both`` always logs per-client. ``server_only`` always suppresses per-client logs.
   * - ``configs.wandb_mode``
     - ``online``
     - WandB mode: ``online`` or ``offline``.
   * - ``configs.wandb_entity``
     - ``""``
     - WandB entity/team.
   * - ``configs.wandb_tags``
     - ``[]``
     - WandB tags (list or comma-separated string). ``seed:<experiment.seed>`` is always appended automatically.
   * - ``configs.track_gen_rewards``
     - ``false``
     - Log per-round and cumulative generalization reward computed from global evaluation error.

Privacy (``privacy``)
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``use_dp``
     - ``false``
     - Enable differential privacy.
   * - ``mechanism``
     - ``laplace``
     - DP backend: ``laplace``, ``gaussian``, or ``opacus``.
   * - ``clip_grad_norm``
     - ``0.0``
     - Gradient clipping norm (disabled when ``0``).
   * - ``clip_norm_type``
     - ``2.0``
     - Norm type used for gradient clipping (e.g., ``2.0`` for L2).
   * - ``kwargs``
     - ``{}``
     - Backend-specific keyword arguments (e.g., Opacus ``noise_multiplier``, ``max_grad_norm``).

Secure Aggregation (``secure_aggregation``)
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``use_sec_agg``
     - ``false``
     - Enable pairwise-masking secure aggregation.
   * - ``mix_coefs``
     - ``uniform``
     - Weighting mode: ``uniform`` or ``sample_ratio``.
