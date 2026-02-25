from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from appfl.sim.algorithm.aggregator import BaseAggregator
from appfl.sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class SwtsScheduler(FedavgScheduler):
    """
    Sliding-window Thompson Sampling for non-contextual local-step adaptation.
    """

    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(
            scheduler_configs=scheduler_configs, aggregator=aggregator, logger=logger
        )
        self.action_space: List[int] = sorted(
            {
                int(x)
                for x in scheduler_configs.get("action_space", [1, 2, 4, 8])
                if int(x) > 0
            }
        )
        if not self.action_space:
            self.action_space = [1]
        self.window_size = max(1, int(scheduler_configs.get("window_size", 50)))
        self.likelihood_variance = max(
            1e-8, float(scheduler_configs.get("likelihood_variance", 1.0))
        )
        self.prior_variance = max(
            1e-8, float(scheduler_configs.get("prior_variance", 1.0))
        )
        self.history: Deque[Tuple[int, float]] = deque(maxlen=self.window_size)
        self.pending_actions: Deque[int] = deque()
        self.prev_pre_val_error: Optional[float] = None
        self.current_round: int = 0
        self.last_selected_action: int = int(self.action_space[0])
        self.last_reward: Optional[float] = None
        seed = scheduler_configs.get("seed", None)
        self._rng = np.random.default_rng(None if seed is None else int(seed))

    def pull(self, round_idx: int) -> int:
        self.current_round = max(1, int(round_idx))
        samples: Dict[int, float] = {}
        for action in self.action_space:
            rewards = [float(r) for a, r in self.history if int(a) == int(action)]
            n = len(rewards)
            reward_sum = float(sum(rewards))
            post_var = 1.0 / (
                (1.0 / self.prior_variance) + (n / self.likelihood_variance)
            )
            post_mean = (post_var / self.likelihood_variance) * reward_sum
            samples[action] = float(
                self._rng.normal(loc=post_mean, scale=np.sqrt(post_var))
            )

        chosen = max(self.action_space, key=lambda a: samples[int(a)])
        chosen = int(chosen)
        self.pending_actions.append(chosen)
        self.last_selected_action = chosen
        return chosen

    def adapt(self, pre_val_error: float) -> Optional[float]:
        current = float(pre_val_error)
        if self.prev_pre_val_error is None:
            self.prev_pre_val_error = current
            self.last_reward = None
            return None

        reward = float(self.prev_pre_val_error - current)
        self.prev_pre_val_error = current
        self.last_reward = float(reward)

        if self.pending_actions:
            action = int(self.pending_actions.popleft())
            self.history.append((action, float(reward)))
        return float(reward)

    def get_bandit_state(self) -> Dict[str, Any]:
        return {
            "name": "swts",
            "round": int(self.current_round),
            "last_action": int(self.last_selected_action),
            "last_reward": self.last_reward,
            "window_size": int(self.window_size),
            "history_size": int(len(self.history)),
            "pending_actions": int(len(self.pending_actions)),
        }
