"""
Evaluate an optimization agent's design points on a Globus Compute endpoint for
compute-heavy evaluation offloading. The endpoint must have a matching ``appfl``
installed, plus whatever the evaluator needs to import.

The agent calls ``evaluator(point, **context)``. :class:`RemoteEvaluator` ships each call
to the endpoint, where :func:`evaluate_on_endpoint` builds the evaluator once per worker
process and reuses it, so per-agent state such as a loaded data shard survives between
calls. It is configured by a ``remote`` block under a node's ``evaluator_configs``::

    remote:
      backend: "globus_compute"
      endpoint_id: "<uuid>"
      timeout: 600                    # seconds to wait for one evaluation; default null waits forever
      evaluator_path: "/abs/path.py"  # required: the evaluator file as the endpoint sees it
      evaluator_name: "get_evaluator" # default: the last function defined in that file
      evaluator_kwargs: {}            # default: {}
      compute_token_env: "APPFL_GC_COMPUTE_TOKEN"  # env vars holding the access tokens;
      openid_token_env: "APPFL_GC_OPENID_TOKEN"    # omit both for the SDK's default login
      user_endpoint_config: {}        # multi-user endpoint template values
      resource_specification: {}      # per-task resources (MPI endpoints)

The block is self-contained: nothing is read from the local ``evaluator_path``,
``evaluator_name`` or ``evaluator_kwargs`` beside it, which only apply without the block.

The endpoint must have a matching ``appfl`` installed, plus whatever the evaluator needs to import.
"""

import os
import json
import time
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
from .remote import GlobusComputeRemote

logger = logging.getLogger(__name__)

_REMOTE_KEYS = {
    "backend",
    "endpoint_id",
    "timeout",
    "evaluator_path",
    "evaluator_name",
    "evaluator_kwargs",
    "compute_token_env",
    "openid_token_env",
    "user_endpoint_config",
    "resource_specification",
}

# Endpoint side: evaluators built in this worker process, keyed by how they were built.
_evaluators: Dict[str, Callable[..., float]] = {}


def evaluate_on_endpoint(
    evaluator_path: str,
    evaluator_name: Optional[str],
    evaluator_kwargs: Dict[str, Any],
    point: Any,
    context: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Runs on the endpoint: evaluate `point` with the agent's evaluation `context` and return
    `(value, seconds spent evaluating)`. The evaluator is built on the first call with these
    arguments and cached afterwards. With no `evaluator_name`, the last function defined in
    `evaluator_path` is used -- resolved here, since the file lives on the endpoint.
    """
    key = json.dumps(
        [evaluator_path, evaluator_name, evaluator_kwargs], sort_keys=True, default=str
    )
    if key not in _evaluators:
        from appfl.decentralized.config import accept_context, load_callable

        if evaluator_name is None:
            from appfl.misc.utils import get_last_function_name

            evaluator_name = get_last_function_name(evaluator_path)
            if evaluator_name is None:
                raise ValueError(
                    f"{evaluator_path} defines no function to evaluate with."
                )
        _evaluators[key] = accept_context(
            load_callable(evaluator_path, evaluator_name)(**evaluator_kwargs)
        )
    start = time.time()
    value = _evaluators[key](point, **context)
    return float(value), time.time() - start


def _evaluate_task(evaluator_path, evaluator_name, evaluator_kwargs, point, context):
    """The function shipped to the endpoint. It is serialized by source and re-executed in
    an empty namespace, so it only imports from the endpoint's installed appfl."""
    from appfl.comm.globus_compute.remote_evaluator import evaluate_on_endpoint

    return evaluate_on_endpoint(
        evaluator_path, evaluator_name, evaluator_kwargs, point, context
    )


class RemoteEvaluator:
    """
    A local instance for an evaluator that runs on a Globus Compute endpoint.

    :param `remote`: The endpoint to evaluate on.
    :param `evaluator_path`: Path of the evaluator file on the endpoint.
    :param `evaluator_name`: Factory in that file that returns the evaluation callable;
        `None` uses the last function defined in the file.
    :param [Optional] `evaluator_kwargs`: Keyword arguments for that factory.
    :param [Optional] `meter`: If given, its `remote_overhead_seconds` accumulates the wall
        time of each call not spent evaluating (queueing, transfer, serialization).
    """

    def __init__(
        self,
        remote: GlobusComputeRemote,
        evaluator_path: str,
        evaluator_name: Optional[str],
        evaluator_kwargs: Optional[Dict[str, Any]] = None,
        meter: Optional[Any] = None,
    ):
        self.remote = remote
        self.evaluator_path = str(evaluator_path)
        self.evaluator_name = None if evaluator_name is None else str(evaluator_name)
        self.evaluator_kwargs = dict(evaluator_kwargs or {})
        self.meter = meter

    def __call__(self, point: Any, **context: Any) -> float:
        """Evaluate `point` remotely, forwarding the agent's context (`eval_idx`, ...)."""
        start = time.time()
        value, endpoint_seconds = self.remote.run(
            _evaluate_task,
            self.evaluator_path,
            self.evaluator_name,
            self.evaluator_kwargs,
            point,
            context,
        )
        if self.meter is not None:
            self.meter.remote_overhead_seconds += max(
                0.0, time.time() - start - endpoint_seconds
            )
        return value

    def close(self):
        self.remote.close()


def create_remote_evaluator(
    evaluator_configs: Mapping, meter: Optional[Any] = None
) -> RemoteEvaluator:
    """
    Build a :class:`RemoteEvaluator` from a node's `evaluator_configs`, which must have a `remote` block.
    """
    if isinstance(evaluator_configs, DictConfig):
        evaluator_configs = OmegaConf.to_container(evaluator_configs, resolve=True)
    remote_configs = dict(evaluator_configs.get("remote", None) or {})
    unknown = set(remote_configs) - _REMOTE_KEYS
    if unknown:
        raise ValueError(
            f"unknown evaluator_configs.remote keys {sorted(unknown)}; "
            f"available: {sorted(_REMOTE_KEYS)}"
        )
    backend = remote_configs.get("backend", "globus_compute")
    if backend != "globus_compute":
        raise ValueError(
            f"unsupported evaluator_configs.remote.backend {backend!r}; "
            "available: ['globus_compute']"
        )
    for key in ("endpoint_id", "evaluator_path"):
        if not remote_configs.get(key, None):
            raise ValueError(f"evaluator_configs.remote.{key} is required.")
    evaluator_name = remote_configs.get("evaluator_name", None) or None
    if evaluator_name is None:
        logger.warning(
            "evaluator_configs.remote.evaluator_name is not set; the endpoint will use the "
            f"last function defined in {remote_configs['evaluator_path']}."
        )

    tokens = {}
    for token in ("compute_token", "openid_token"):
        env_name = remote_configs.get(f"{token}_env", None)
        if env_name is None:
            tokens[token] = None
        elif os.environ.get(env_name):
            tokens[token] = os.environ[env_name]
        else:
            raise ValueError(
                f"evaluator_configs.remote.{token}_env names {env_name!r}, "
                "but that environment variable is not set."
            )

    timeout = remote_configs.get("timeout", None)
    remote = GlobusComputeRemote(
        endpoint_id=remote_configs["endpoint_id"],
        timeout=None if timeout is None else float(timeout),
        compute_token=tokens["compute_token"],
        openid_token=tokens["openid_token"],
        user_endpoint_config=remote_configs.get("user_endpoint_config", None),
        resource_specification=remote_configs.get("resource_specification", None),
    )
    return RemoteEvaluator(
        remote,
        evaluator_path=remote_configs["evaluator_path"],
        evaluator_name=evaluator_name,
        evaluator_kwargs=remote_configs.get("evaluator_kwargs", None) or {},
        meter=meter,
    )
