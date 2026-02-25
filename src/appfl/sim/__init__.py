"""appfl[sim]: simulation-focused FL package."""


def run_distributed(config, backend: str = "gloo") -> None:
    from appfl.sim.runner import run_distributed as _run_distributed

    _run_distributed(config, backend=backend)


def run_serial(config) -> None:
    from appfl.sim.runner import run_serial as _run_serial

    _run_serial(config)


__all__ = ["run_distributed", "run_serial"]
