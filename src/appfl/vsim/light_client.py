"""Lightweight virtual clients that share one ClientAgent per process.

A ClientAgent owns a model, a trainer and copies of the model state. A process
hosting hundreds of them holds gigabytes (about 185 MB per ResNet-18 client), and
APPFL's frequent ``gc.collect()`` calls walk all of them on every task. In a
simulation, though, every task starts by loading the global model over the local
one, and VanillaTrainer builds a fresh optimizer in each ``train()``, so between
tasks a client carries only its data and its round counter.

:class:`SharedAgentPool` therefore keeps one ClientAgent, the engine, and per client
only its datasets. A :class:`LightClient` binds its datasets and round counter to
the engine before each call, so any number of clients train on one model, one
trainer configuration and one device. The datasets are loaded once per process, the
first time a client is requested, through the same code path as
``ClientAgent._load_data``. Because a light client can be built anywhere at the
cost of loading its data, it is also what lets dynamic placement train any client
on any MPI worker.

Results are identical to full ClientAgents as long as the client has no other state:
the global model must overwrite every entry of the local model, and the trainer must
keep nothing between tasks but its round counter (VanillaTrainer without Opacus
privacy accounting). The pool checks both.
"""

import types
from typing import Any, Callable, Dict, Optional, Tuple


def load_client_datasets(client_agent_config) -> Tuple[Any, Any]:
    """
    Load ``(train_dataset, val_dataset)`` exactly as a ClientAgent would.

    Runs ``ClientAgent._load_data`` on the configuration alone, without building the
    agent's model, trainer or logger.

    :param client_agent_config: A client configuration with ``data_configs``.
    :return: The client's training and validation datasets.
    """
    from appfl.agent import ClientAgent

    holder = types.SimpleNamespace(client_agent_config=client_agent_config)
    ClientAgent._load_data(holder)
    return holder.train_dataset, holder.val_dataset


def _stateless_trainers():
    from appfl.algorithm.trainer import VanillaTrainer

    return (VanillaTrainer,)


def _shared_logger(client_agent_config):
    """
    A client logger for the engine whose tables name the client of each line.

    The engine's own logger carries the id of the client it was built for, which
    would mislabel every other client's lines.
    """
    from appfl.logger import ClientAgentFileLogger

    class SharedLogger(ClientAgentFileLogger):
        cid = ""

        def log_title(self, titles):
            super().log_title(["Client", *titles])

        def set_title(self, titles):
            # Clients may log different columns (e.g. one without validation
            # data), so each sets its own.
            self.titles = ["Client", *titles]

        def log_content(self, contents):
            if isinstance(contents, dict):
                contents = {"Client": self.cid, **contents}
            else:
                contents = [self.cid, *contents]
            super().log_content(contents)

    train = client_agent_config.get("train_configs", {}) or {}
    return SharedLogger(
        file_dir=train.get("logging_output_dirname", "./output"),
        file_name=train.get("logging_output_filename", "result"),
        experiment_id=client_agent_config.get("experiment_id", ""),
    )


class SharedAgentPool:
    """
    Serve any number of virtual clients from one ClientAgent per process.

    Use an instance as the ``make_client`` of :func:`appfl.vsim.mpi.run_worker`, or
    call :meth:`client` to build client lists for an in-process driver. The engine
    logs through one logger whose tables name each line's client.

    :param make_agent: ``make_agent(cid, context, device)`` returning a configured
        ClientAgent for client ``cid``. It is called once, for the first client
        requested; that agent's model, loss, metric, trainer configuration and
        compressor then serve every client.
    :param load_data: ``load_data(cid, context, device)`` returning
        ``(train_dataset, val_dataset)`` for client ``cid``, e.g. through
        :func:`load_client_datasets`. Called once per client.
    :param stateless_trainers: Trainer classes known to keep no per-client state
        between tasks other than ``round``. Defaults to ``(VanillaTrainer,)``; the
        trainer's exact class must be listed.
    """

    def __init__(
        self,
        make_agent: Callable,
        load_data: Callable,
        stateless_trainers: Optional[Tuple[type, ...]] = None,
    ):
        self._make_agent = make_agent
        self._load_data = load_data
        self._trainers = stateless_trainers
        self.agent = None
        self._origin = None  # (context, device) the engine was built with
        self._data: Dict[str, Tuple[Any, Any]] = {}
        self._bound = None
        self._state_keys = None
        self.stats = {"data_loads": 0, "binds": 0}

    def __call__(self, cid: str, context=None, device=None) -> "LightClient":
        """``make_client``: return a new client handle for ``cid``."""
        return self.client(cid, context, device)

    def client(self, cid: str, context=None, device=None) -> "LightClient":
        """
        Return a new handle on client ``cid``, loading its data if needed.

        Every handle starts with a round counter of zero; handles of one client
        share its datasets. A different ``context`` or ``device`` than the engine
        was built with (a new run with another configuration) starts over.
        """
        if self.agent is not None and self._origin != (context, device):
            self.agent = self._origin = self._bound = self._state_keys = None
            self._data.clear()
        if cid not in self._data:
            if self.agent is None:
                self._start(cid, context, device)
            else:
                self._data[cid] = tuple(self._load_data(cid, context, device))
                self.stats["data_loads"] += 1
        return LightClient(self, cid)

    def _start(self, cid: str, context, device) -> None:
        """Build the engine from client ``cid``, whose data comes with it."""
        agent = self._make_agent(cid, context, device)
        trainer = getattr(agent, "trainer", None)
        allowed = self._trainers or _stateless_trainers()
        if type(trainer) not in allowed:
            raise TypeError(
                f"SharedAgentPool reuses one trainer for every client, which is exact "
                f"only for trainers that keep no per-client state between tasks; got "
                f"{type(trainer).__name__}, expected one of "
                f"{[t.__name__ for t in allowed]}. Train such clients with full "
                f"ClientAgents (and static placement or pinned clients under MPI), "
                f"or list the trainer in `stateless_trainers` if it is stateless."
            )
        configs = trainer.train_configs
        if (
            configs.get("use_dp", False)
            and configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            raise ValueError(
                "Opacus differential privacy accounts the privacy budget per client "
                "across tasks, so these clients cannot share a trainer. Train them "
                "with full ClientAgents (and static placement or pinned clients "
                "under MPI)."
            )
        config = getattr(agent, "client_agent_config", None)
        if config is not None:
            agent.logger = _shared_logger(config)
        self.agent = agent
        self._origin = (context, device)
        self._data[cid] = (agent.train_dataset, agent.val_dataset)
        self.stats["data_loads"] += 1

    def bind(self, client: "LightClient"):
        """Install ``client``'s data and round counter in the engine; return it."""
        agent = self.agent
        if self._bound is not client:
            agent.client_id = client.cid
            if hasattr(agent.logger, "cid"):
                agent.logger.cid = client.cid
            agent.train_dataset, agent.val_dataset = client.data
            # A new trainer around the shared model, loss and metric picks up the
            # client's datasets through APPFL's own construction path.
            agent.trainer = None
            agent._load_trainer()
            self._bound = client
            self.stats["binds"] += 1
        agent.trainer.round = client.round
        return agent

    def check_covers(self, params) -> None:
        """Raise if ``params`` leaves some entry of the shared model untouched."""
        if self._state_keys is None:
            model = getattr(self.agent, "model", None)
            if model is None:
                return
            self._state_keys = frozenset(model.state_dict())
        if not hasattr(params, "keys"):
            return
        missing = self._state_keys.difference(params.keys())
        if missing:
            raise RuntimeError(
                f"The global model does not cover {sorted(missing)[:5]} of the local "
                f"model. A client would keep its own values of those entries between "
                f"tasks, which a shared model cannot. Train such clients with full "
                f"ClientAgents (and static placement or pinned clients under MPI)."
            )


class LightClient:
    """
    A virtual client that trains on its pool's shared ClientAgent.

    It holds its id, its datasets and its round counter, and offers the methods the
    vsim drivers and MPI workers use: ``get_id``, ``get_sample_size``,
    ``expected_local_steps``, ``load_parameters``, ``train``, ``get_parameters``,
    and ``set_round``, through which an MPI worker passes on the client's task
    count when its earlier tasks ran elsewhere.
    """

    __slots__ = ("pool", "cid", "data", "round")

    def __init__(self, pool: SharedAgentPool, cid: str):
        self.pool = pool
        self.cid = cid
        self.data = pool._data[cid]
        self.round = 0

    def get_id(self) -> str:
        return self.cid

    def get_sample_size(self) -> int:
        return len(self.data[0])

    def set_round(self, value: int) -> None:
        """Set how many tasks this client trained before the next one."""
        self.round = int(value)

    def expected_local_steps(self) -> int:
        from .mpi.worker import expected_local_steps

        return expected_local_steps(self.pool.bind(self))

    def load_parameters(self, params) -> None:
        agent = self.pool.bind(self)
        self.pool.check_covers(params)
        agent.load_parameters(params)

    def train(self, **kwargs) -> None:
        agent = self.pool.bind(self)
        agent.train(**kwargs)
        self.round = agent.trainer.round

    def get_parameters(self):
        agent = self.pool.bind(self)
        result = agent.get_parameters()
        # The caller owns the local model now; the engine need not keep it.
        agent.trainer.__dict__.pop("model_state", None)
        return result
