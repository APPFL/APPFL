Trajectory consistency-based stopping
====================================

``TCSTrainer`` adapts local SGD effort by comparing the average update with a
reference formed during the base steps. ``TCSAggregator`` combines the resulting
models with sample weights and work normalization. Use them with ``SyncScheduler``.
Paper resources and the MNIST reproduction are maintained at https://github.com/APPFL/TCS.

Configuration
-------------

Use the existing server/client example scripts with
``examples/resources/configs/mnist/server_tcs.yaml``. The example retains APPFL's
existing CNN; the standalone paper reproduction uses a two-hidden-layer MLP.

.. code-block:: yaml

   client_configs:
     train_configs:
       trainer: TCSTrainer
       mode: step
       num_base_steps: 10
       num_local_steps: 100
       kappa: 0.5
       optim: SGD
       optim_args: {lr: 0.1}
   server_configs:
     scheduler: SyncScheduler
     aggregator: TCSAggregator

Step mode is recommended. ``num_local_steps`` is a cap, not a mandatory count.
For epoch mode, set ``mode: epoch`` and ``num_local_epochs`` to the desired
integer maximum (e.g. 2, 5 or 10). The base is exactly one dataloader pass;
checks after each later batch permit fractional epochs such as 1.2 or 1.8.
The first global round executes only the base, not the maximum.

Contract
--------

The average direction at step h is ``d = (initial - current) / (lr * h)``.
Freeze ``r`` at the base length; continue only while
``norm(d) - norm(d-r) - kappa*norm(r) > 0`` and below the cap.
Keep the update at the first failed check. A zero reference stops at the base.
No additional forward evaluation or RPC is used for stopping.

The server computes ``p_i = n_i / sum(n_j)`` over participating clients,
``Hbar = sum(p_i * H_i)``, then adds
``Hbar * sum(p_i * (local_i - global) / H_i)`` to the global parameters.
Equal counts recover sample-weighted FedAvg.

Client metadata includes ``round``, ``num_steps``, ``num_samples`` and
``learning_rate``. ``SyncScheduler`` forwards these dictionaries without changes.
The initial and updated global models include the global ``round`` in response
metadata. Pass that metadata to ``client_agent.train(**metadata)`` so a client's
first participation in a later global round does not repeat round-zero behavior.
The existing gRPC client example already passes metadata to training.

Only plain, zero-momentum SGD with a common positive learning rate within each
round is supported. Weight decay, DP, clipping and gradient payloads are rejected.
An optional ``local_steps`` runtime cap may tighten the configured maximum; a cap
below the base length is rejected. The experiment driver owns client sampling,
work-budget accounting and learning-rate scheduling.

The stopping signal uses trainable parameters. Floating buffers are separately
sample-weighted; differing nonfloating buffers are rejected. Local validation is
not part of this trainer; evaluate outside the stopping loop. Full-model payloads
use APPFL's existing transport; streamed aggregation of partial model chunks is
not supported. All input validation precedes committing the aggregate.
