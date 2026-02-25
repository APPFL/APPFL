import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from appfl.sim.algorithm.aggregator import BaseAggregator
from appfl.sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class SwucbScheduler(FedavgScheduler):
    """
    Sliding-window UCB for non-contextual local-step adaptation.
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
        self.window_size = max(1, int(scheduler_configs.get("window_size", 5)))
        self.exploration_alpha = float(scheduler_configs.get("exploration_alpha", 0.1))
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
        counts: Dict[int, int] = {a: 0 for a in self.action_space}
        sums: Dict[int, float] = {a: 0.0 for a in self.action_space}
        for action, reward in self.history:
            counts[action] += 1
            sums[action] += float(reward)

        missing = [a for a in self.action_space if counts[a] == 0]
        if missing:
            chosen = int(self._rng.choice(missing))
        else:
            chosen = None
            best_score = float("-inf")
            for action in self.action_space:
                n = max(1, counts[action])
                mean_reward = sums[action] / n
                ucb = self.exploration_alpha * math.sqrt(
                    max(0.0, math.log(float(self.current_round))) / n
                )
                score = mean_reward + ucb
                if score > best_score:
                    best_score = score
                    chosen = int(action)
            if chosen is None:
                chosen = self._rng.choice(self.action_space)

        self.pending_actions.append(chosen)
        self.last_selected_action = int(chosen)
        return int(chosen)

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
            "name": "swucb",
            "round": int(self.current_round),
            "last_action": int(self.last_selected_action),
            "last_reward": self.last_reward,
            "window_size": int(self.window_size),
            "history_size": int(len(self.history)),
            "pending_actions": int(len(self.pending_actions)),
        }
