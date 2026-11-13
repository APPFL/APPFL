import abc
from typing import Dict, Iterable, Union, OrderedDict, Tuple


class BaseAggregator:
    def set_client_sample_size(self, client_id: Union[str, int], sample_size: int):
        """Set the sample size of a client"""
        if not hasattr(self, "client_sample_size"):
            self.client_sample_size = {}
        self.client_sample_size[client_id] = sample_size

    def _warn_missing_sample_sizes(self, client_ids: Iterable[Union[str, int]]) -> None:
        """
        Warn once per aggregator if `client_weights_mode` is "sample_size" but some of
        `client_ids` have no registered sample size, so they are weighted as in "equal"
        mode. A no-op unless the subclass defines `client_weights_mode` and `logger`.
        """
        if (
            getattr(self, "client_weights_mode", None) != "sample_size"
            or getattr(self, "_sample_size_warning_issued", False)
            or getattr(self, "logger", None) is None
        ):
            return
        registered = getattr(self, "client_sample_size", {})
        # repr() so a key-type mismatch (registered 0, aggregated "0") is visible.
        missing = [repr(cid) for cid in client_ids if cid not in registered]
        if not missing:
            return
        self._sample_size_warning_issued = True
        shown = ", ".join(missing[:5])
        if len(missing) > 5:
            shown += f" and {len(missing) - 5} more"
        self.logger.warning(
            f"{type(self).__name__}: client_weights_mode is 'sample_size', but no "
            f"sample size is registered for client(s) {shown}, so they are weighted "
            "as in 'equal' mode instead. Register every client's sample size with "
            "`ServerAgent.set_sample_size` (or `appfl.vsim.register_sample_sizes` "
            "for the virtual-time simulator). (Logged once; later clients without a "
            "sample size are not reported.)"
        )

    @abc.abstractmethod
    def aggregate(
        self, *args, **kwargs
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Aggregate local model(s) from clients and return the global model
        """
        pass

    @abc.abstractmethod
    def get_parameters(
        self, **kwargs
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return global model parameters"""
        pass
