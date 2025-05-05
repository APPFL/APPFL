from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class ClientTask:
    task_id: str = ""
    task_name: str = ""
    client_id: str = ""
    pending: bool = True
    success: bool = False
    start_time: float = -1
    end_time: float = -1
    log: Optional[Dict] = field(default_factory=dict)
    parameters: Optional[Dict] = field(default_factory=dict)
    failure: bool = False
    task_execution_start_time: float = -1
    task_execution_finish_time: float = -1
    result: Any = None
    est_finish_time: float = -1
    spin_up_time: float = -1
    task_execution_time: float = -1
    is_instance_alive: bool = False
