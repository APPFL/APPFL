try:
    from appfl.sim.algorithm.aggregator.fedavg_aggregator import (
        FedavgAggregator as _FedavgBase,
    )
except ImportError:  # pragma: no cover
    from appfl.sim.algorithm.aggregator.fedavg_aggregator import (
        FedAvgAggregator as _FedavgBase,
    )


class SwtsAggregator(_FedavgBase):
    pass
