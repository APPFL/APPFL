import threading
import time


class PayloadStore:
    """Payloads published by one node, by round, as bytes ready to send.

    Serialized once on publish rather than once per requester, and kept for a bounded number
    of rounds: a neighbor may legitimately be one round behind -- a node publishes `r` before
    collecting `r`, so by the time a slower neighbor asks it may already hold `r + 1` -- but no
    further, because publishing `r + 1` required collecting that neighbor's `r`.

    Used by a node serving its own payloads and by a relay holding one of these per publisher,
    so the retention rule and the blocking read are defined once for both.
    """

    def __init__(self, history: int = 2):
        self.history = history
        self._rounds: dict[int, bytes] = {}
        self._latest = -1
        self._condition = threading.Condition()

    def publish(self, round_id: int, data: bytes) -> None:
        with self._condition:
            self._rounds[round_id] = data
            self._latest = max(self._latest, round_id)
            for stale in sorted(self._rounds)[: -self.history]:
                del self._rounds[stale]
            self._condition.notify_all()

    def get(self, round_id: int, timeout: float) -> bytes:
        """Block until ``round_id`` is published, then return it.

        This wait is the round barrier: a node cannot complete a round until everyone it
        collects from has published that round, because this is what it is waiting on.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while round_id not in self._rounds:
                if round_id < self._latest:
                    raise KeyError(
                        f"round {round_id} was asked for but this publisher has already moved "
                        f"past it (now at {self._latest}, keeping {self.history}). The "
                        f"requester has fallen further behind than the protocol allows."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(remaining):
                    raise TimeoutError(
                        f"round {round_id} was not published within {timeout:.0f}s"
                    )
            return self._rounds[round_id]
