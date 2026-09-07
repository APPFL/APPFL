from dataclasses import dataclass, field


@dataclass
class ClientTask:
    task_id: str = ""
    task_name: str = ""
    client_id: str = ""
    pending: bool = True
    success: bool = False
    start_time: float = -1
    end_time: float = -1
    log: dict | None = field(default_factory=dict)
