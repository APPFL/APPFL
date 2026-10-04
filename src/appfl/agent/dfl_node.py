import importlib
import threading
from collections import OrderedDict as OrderedDictType
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, OrderedDict, Union

import torch
from omegaconf import DictConfig, OmegaConf

from appfl.agent.client import ClientAgent
from appfl.algorithm.aggregator import BaseAggregator
from appfl.decentralized.neighbor import Neighbors, resolve_neighbors


class DFLNodeAgent(ClientAgent):
    """
    `DFLNodeAgent`
    This is the agent class for a Decentralized FL (DFL) node. Unlike centralized federated
    learning, a DFL node acts both as a server and as a client: it trains a local model on its
    own data, requests local models from its neighbor nodes to update that model, and in turn
    serves its own local model to the neighbors that request it.

    It subclasses :class:`~appfl.agent.ClientAgent`, so the model, loss, metric, dataset,
    trainer and compressor are loaded from configuration in exactly the same way as for a
    centralized FL client -- a DFL node is an FL client, plus the aggregation duty that a
    centralized run would leave to the server. Everything a `ClientAgent` supports therefore
    works unchanged.

    What this class adds on top of `ClientAgent`:

    - an **aggregator**, since there is no server to hold one;
    - **round-tagged model publication**, so a neighbor asking for round `r` receives the
      model from round `r` and not whatever this node happens to hold at the time;
    - **connection bookkeeping**, so a node can stop serving once every neighbor is done.

    :param dfl_node_agent_config: configurations for the DFL node agent. A superset of
        `ClientAgentConfig`: it additionally takes `neighbors`, `aggregator`,
        `aggregator_kwargs` and `num_epochs`.
    :param neighbors: this node's resolved neighbor view. **Normally omitted**, and read from the
        config's `neighbors` block instead. A simulation launcher passes it explicitly, having
        resolved `from_topo` against a topology it built over the whole federation --
        which is the only place a global graph exists, and never inside a node.
    """

    #: How many past rounds of published parameters to keep. A neighbor can legitimately be
    #: one round behind -- a node publishes round `r` before pulling round `r`, so by the time
    #: a slow neighbor asks, this node may already have published `r + 1` -- but no more than
    #: one, because publishing `r + 1` requires having pulled that neighbor's round `r`.
    _published_history: int = 2

    def __init__(
        self,
        dfl_node_agent_config: DictConfig = DictConfig({}),
        neighbors: Optional[Neighbors] = None,
        **kwargs,
    ) -> None:
        # Set before `super().__init__()`: it calls `_create_logger`, which calls `get_id`,
        # which reads the config -- so the alias has to exist before the base constructor runs.
        self.dfl_node_agent_config = dfl_node_agent_config
        super().__init__(client_agent_config=dfl_node_agent_config, **kwargs)
        self.neighbors = (
            neighbors
            if neighbors is not None
            else resolve_neighbors(
                dfl_node_agent_config.get("neighbors", None), self.get_id()
            )
        )
        self._load_aggregator()

        self._round = -1  # no round has been published yet
        self._published: "OrderedDictType[int, Any]" = OrderedDictType()
        self._pending: Dict[int, List[Future]] = {}
        self._model_params_lock = threading.Lock()

        self._closed_neighbors = set()
        self._close_connection_lock = threading.Lock()
        self._num_neighbors = self._resolve_num_neighbors()

    # -- identity ---------------------------------------------------------------------

    def get_id(self) -> str:
        """Return a unique node id, for neighbors to distinguish this node.

        `node_id` is the DFL spelling; `client_id` is inherited from `ClientAgent` and is
        accepted so an existing FL client config can be used unchanged.
        """
        if not hasattr(self, "client_id"):
            if hasattr(self.dfl_node_agent_config, "node_id"):
                self.client_id = str(self.dfl_node_agent_config.node_id)
                return self.client_id
        return super().get_id()

    @property
    def node_id(self) -> str:
        return self.get_id()

    # -- the client half: train, then publish -----------------------------------------

    def train(self, **kwargs) -> None:
        """Train the local model, then publish the result for this round.

        Publication is what unblocks neighbors waiting on :meth:`get_parameters`. It happens
        immediately after training and before this node pulls anything, so a federation cannot
        deadlock with every node waiting for every other node to go first.
        """
        super().train(**kwargs)
        self.publish_parameters()

    def publish_parameters(self, round_id: Optional[int] = None) -> None:
        """Make the current local model available to neighbors as round ``round_id``."""
        params = self.trainer.get_parameters()
        if isinstance(params, tuple):
            params = params[0]
        # Detach from the trainer's own tensors. Aggregation loads new parameters into the
        # model in place, and a neighbor that has not collected yet must still receive this
        # round's model rather than the mixed one that replaced it.
        params = {
            name: (tensor.detach().clone() if torch.is_tensor(tensor) else tensor)
            for name, tensor in params.items()
        }
        with self._model_params_lock:
            self._round = self._round + 1 if round_id is None else round_id
            self._published[self._round] = params
            while len(self._published) > self._published_history:
                self._published.popitem(last=False)
            for future in self._pending.pop(self._round, []):
                future.set_result(params)

    # -- the server half: serve this node's model to a neighbor ------------------------

    def get_parameters(
        self,
        round_id: Optional[int] = None,
        blocking: bool = True,
        requester_id: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> Union[Dict, OrderedDict, Future]:
        """Return this node's local model parameters for a given round.

        :param round_id: which round's model is wanted. `None` means "whatever is current",
            which is only safe when the caller does not care about round alignment.
        :param blocking: whether to wait for the round to be published, or to return a
            `Future` that resolves when it is.
        :param requester_id: the peer asking, checked against `send_to`. `None` is a local
            call and is never checked.

        Waiting is the point. In synchronous DFL a node must average its neighbors' round-`r`
        models, not whatever they last happened to publish; without this, a fast node silently
        trains on stale neighbors and the run stops being reproducible -- and stops matching
        centralized FedAvg on a fully connected graph.
        """
        # The declaration lives on `Neighbors`, so in-process and networked callers apply
        # the same rule rather than each keeping its own opinion of it.
        if requester_id is not None and not self.neighbors.may_serve(str(requester_id)):
            raise PermissionError(
                f"node {self.get_id()} does not serve {requester_id}: it is not in this "
                f"node's `send_to` list ({self.neighbors.send_to})."
            )
        with self._model_params_lock:
            wanted = round_id
            if wanted is None:
                wanted = self._round if self._round >= 0 else 0

            if wanted in self._published:
                return self._published[wanted]
            if wanted < self._round:
                raise ValueError(
                    f"node {self.get_id()} was asked for round {wanted} but has already "
                    f"moved past it (current round {self._round}, keeping "
                    f"{self._published_history}). The requesting neighbor has fallen further "
                    f"behind than the synchronous protocol allows."
                )

            future = Future()
            self._pending.setdefault(wanted, []).append(future)
        return future.result() if blocking else future

    # -- the server half: aggregate ----------------------------------------------------

    def aggregate_parameters(
        self,
        neighbor_models: Union[
            Dict[str, Union[Dict, OrderedDict]], List[Union[Dict, OrderedDict]]
        ],
        **kwargs,
    ) -> Union[Dict, OrderedDict]:
        """Mix this node's model with its neighbors' and load the result into the trainer.

        This is the step a centralized run performs on the server. Here it happens on every
        node, over its own neighborhood only -- which is the whole difference between DFL and
        FL, and the reason a fully connected graph collapses the two.
        """
        if self._num_neighbors is None:
            self._num_neighbors = len(neighbor_models)
        local_model = self.get_parameters(blocking=True)
        new_model = self.aggregator.aggregate(
            local_model,
            neighbor_models,
            self.get_id(),
            mixing_weights=self._mixing_weights(neighbor_models),
            **kwargs,
        )
        with self._model_params_lock:
            self.trainer.load_parameters(new_model)
        return new_model

    def _mixing_weights(self, neighbor_models) -> Optional[Dict[str, float]]:
        """``pi_ij`` for this node and each neighbor, or `None` to let the aggregator average.

        Available whenever the neighbor models are keyed by node id. On a regular graph the
        weights are uniform and this changes nothing; on an irregular one it is the difference
        between a correct mixing matrix and an average that quietly over-weights low-degree
        neighbors.
        """
        if not isinstance(neighbor_models, dict) or not self.neighbors.recv_from:
            return None
        return self.neighbors.mixing_weights(self.get_id())

    # -- connection bookkeeping --------------------------------------------------------

    def close_connection(self, neighbor_id: Union[str, int]) -> None:
        """Record that ``neighbor_id`` is finished and will not request anything further."""
        with self._close_connection_lock:
            self._closed_neighbors.add(str(neighbor_id))

    def server_terminated(self) -> bool:
        """Whether every neighbor has closed its connection, so this node may stop serving."""
        if self._num_neighbors is None:
            return False
        with self._close_connection_lock:
            return len(self._closed_neighbors) >= self._num_neighbors

    # -- loading -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """Seed before building the model, so every node starts from identical parameters.

        Centralized FL gets this for free: the server builds one model and ships it to every
        client. In DFL there is no server, so each node builds its own -- and without a shared
        seed the federation starts from `n` different random initializations. It will still
        converge, but the run is then not comparable to a centralized one, and consensus takes
        rounds that should not have been needed.

        Set `model_configs.seed` to `null` to opt out and let each node initialize randomly.
        """
        model_configs = getattr(self.client_agent_config, "model_configs", None)
        seed = model_configs.get("seed", 42) if model_configs is not None else None
        if seed is not None:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        super()._load_model()

    def _load_aggregator(self) -> None:
        """Instantiate the aggregator named in the config, the way the server agent does."""
        aggregator_name = self.dfl_node_agent_config.get(
            "aggregator", "DFLNodeFedAvgAggregator"
        )
        aggregator_module = importlib.import_module("appfl.algorithm.aggregator")
        if not hasattr(aggregator_module, aggregator_name):
            raise ValueError(f"Invalid aggregator name: {aggregator_name}")
        self.aggregator: BaseAggregator = getattr(aggregator_module, aggregator_name)(
            self.model,
            OmegaConf.create(self.dfl_node_agent_config.get("aggregator_kwargs", {})),
            self.logger,
        )

    def _resolve_num_neighbors(self) -> Optional[int]:
        """How many peers will pull from this node, so it knows when it may stop serving.

        That is `send_to`, not `recv_from`: the nodes this one *serves* are the ones whose
        close-connection calls it is waiting for. On a symmetric graph the two coincide, which
        is why the distinction only shows up once someone writes a directed one.
        """
        if self.neighbors.send_to:
            return len(self.neighbors.send_to)
        configured = self.dfl_node_agent_config.get("num_neighbors", None)
        return int(configured) if configured is not None else None
