from dataclasses import dataclass


@dataclass
class ClientInfo:
    client_id: str
    budget: float = 0.0
    instance_type: str = 't3.medium'
    sample_size: int = 0
    batch_size: int = 0
    steps_per_epoch: int = 0

    time_per_step: float = 0.0
    train_time_overhead_sec: float = 0.0
    initial_train_time_overhead_sec: float = 0.0
    spot_price_per_hr: float = 0.0
    est_spinup_time: float = -1
    est_time_per_epoch: float = -1
    est_cost_per_epoch: float = 0
    alpha: float = 0.7

    inactive: bool = False
    instance_alive: bool = False
    instance_triggered: bool = False
    instance_triggered_at: float = -1

    total_cost_incur: float = 0.0
    client_total_time: int = 0

    on_demand_price_per_hr: float = 0.0
    terminate_savings: float = 0.0
    on_demand_total_cost: float = 0.0
    spot_only_total_cost: float = 0.0
