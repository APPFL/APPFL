"""
DecentralizedTrainer: Client-side trainer for decentralized learning.

Each client performs standard local SGD training independently, then
sends its parameters to the server. The server performs the consensus
(topology-weighted averaging) step and returns a per-client model.

Inherits VanillaTrainer without modification — no special BatchNorm
handling is needed because decentralized consensus averaging does not
alter activation distributions the way DIMAT's activation-matching merge
does.
"""

from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer


class DecentralizedTrainer(VanillaTrainer):
    """
    Thin wrapper around VanillaTrainer for decentralized learning.

    The trainer performs standard local SGD (or any configured optimizer)
    for ``num_local_epochs`` epochs or ``num_local_steps`` steps before
    uploading parameters to the server for consensus aggregation.

    No architectural changes from VanillaTrainer are required; this class
    exists primarily to document intent and allow easy future extension
    (e.g., local momentum correction, gradient tracking).
    """

    pass
