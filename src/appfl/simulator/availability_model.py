"""Availability/dropout models for the virtual-time async FL simulator (v2).

Four dispatch-level models (checked before dispatching a client):
  PermanentDropout   — client exits forever with probability p per dispatch attempt
  SessionDropout     — cyclic active/inactive periods per client
  CorrelatedDropout  — group of clients fail simultaneously

One completion-level model (checked after training, before aggregation):
  TimeoutModel       — server discards results slower than a deadline
"""

import random as _random


class AvailabilityModel:
    """Base: all clients always available."""

    def available(self, cid, vtime):
        return True


class PermanentDropout(AvailabilityModel):
    def __init__(self, drop_prob=0.0, seed=42):
        self.drop_prob = drop_prob
        self.dropped = set()
        self.rng = _random.Random(seed)

    def available(self, cid, vtime):
        if cid in self.dropped:
            return False
        if self.rng.random() < self.drop_prob:
            self.dropped.add(cid)
            return False
        return True


class SessionDropout(AvailabilityModel):
    def __init__(self, active_duration=60.0, inactive_duration=30.0,
                 phase_noise=0.2, seed=42):
        self.active_dur = active_duration
        self.inactive_dur = inactive_duration
        self.phase_noise = phase_noise
        self.rng = _random.Random(seed)
        self.phases = {}

    def available(self, cid, vtime):
        if cid not in self.phases:
            cycle = self.active_dur + self.inactive_dur
            offset = self.rng.uniform(0, cycle)
            if offset < self.active_dur:
                state = "active"
                next_t = self.active_dur - offset
            else:
                state = "inactive"
                next_t = cycle - offset
            next_t *= (1 + self.rng.uniform(-self.phase_noise, self.phase_noise))
            self.phases[cid] = (state, next_t)

        state, next_t = self.phases[cid]
        while vtime >= next_t:
            state = "inactive" if state == "active" else "active"
            dur = self.active_dur if state == "active" else self.inactive_dur
            dur *= (1 + self.rng.uniform(-self.phase_noise, self.phase_noise))
            next_t += dur
            self.phases[cid] = (state, next_t)

        return state == "active"


class CorrelatedDropout(AvailabilityModel):
    def __init__(self, groups, failure_prob=0.05, failure_duration=30.0, seed=42):
        self.groups = groups
        self.failure_prob = failure_prob
        self.failure_duration = failure_duration
        self.rng = _random.Random(seed)
        self.group_failures = {}

    def available(self, cid, vtime):
        group = self.groups.get(cid, "default")

        if group in self.group_failures:
            if vtime < self.group_failures[group]:
                return False
            del self.group_failures[group]

        if self.rng.random() < self.failure_prob:
            self.group_failures[group] = vtime + self.failure_duration
            return False

        return True


class TimeoutModel:
    def __init__(self, timeout_seconds=None, timeout_quantile=None):
        self.timeout = timeout_seconds
        self.quantile = timeout_quantile

    def should_discard(self, duration, all_durations):
        if self.timeout is not None:
            return duration > self.timeout
        if self.quantile is not None and all_durations:
            sorted_d = sorted(all_durations)
            idx = min(int(len(sorted_d) * self.quantile), len(sorted_d) - 1)
            return duration > sorted_d[idx]
        return False


def build_availability(avail_cfg, client_ids, seed):
    """Factory: create (availability_model, timeout_model) from config dict."""
    if not avail_cfg:
        return None, None

    mode = avail_cfg.get("mode", "none")
    if isinstance(avail_cfg, str):
        mode = avail_cfg
        avail_cfg = {}

    timeout_model = None
    tc = avail_cfg.get("timeout", {})
    if tc:
        timeout_model = TimeoutModel(
            timeout_seconds=tc.get("timeout_seconds"),
            timeout_quantile=tc.get("timeout_quantile"),
        )

    availability_model = None
    if mode == "permanent":
        pc = avail_cfg.get("permanent", {})
        availability_model = PermanentDropout(
            drop_prob=pc.get("drop_prob", 0.02), seed=seed,
        )
    elif mode == "session":
        sc = avail_cfg.get("session", {})
        availability_model = SessionDropout(
            active_duration=sc.get("active_duration", 300.0),
            inactive_duration=sc.get("inactive_duration", 600.0),
            phase_noise=sc.get("phase_noise", 0.2),
            seed=seed,
        )
    elif mode == "correlated":
        cc = avail_cfg.get("correlated", {})
        num_groups = cc.get("num_groups", 3)
        groups = {cid: i % num_groups for i, cid in enumerate(client_ids)}
        availability_model = CorrelatedDropout(
            groups=groups,
            failure_prob=cc.get("failure_prob", 0.05),
            failure_duration=cc.get("failure_duration", 30.0),
            seed=seed,
        )
    elif mode == "composite":
        models = []
        if "permanent" in avail_cfg:
            pc = avail_cfg["permanent"]
            models.append(PermanentDropout(pc.get("drop_prob", 0.02), seed))
        if "session" in avail_cfg:
            sc = avail_cfg["session"]
            models.append(SessionDropout(
                sc.get("active_duration", 300.0),
                sc.get("inactive_duration", 600.0),
                sc.get("phase_noise", 0.2),
                seed + 1,
            ))
        if "correlated" in avail_cfg:
            cc = avail_cfg["correlated"]
            num_groups = cc.get("num_groups", 3)
            groups = {cid: i % num_groups for i, cid in enumerate(client_ids)}
            models.append(CorrelatedDropout(groups, cc.get("failure_prob", 0.05),
                                            cc.get("failure_duration", 30.0), seed + 2))
        if models:
            class _Composite(AvailabilityModel):
                def __init__(self, ms):
                    self.models = ms
                def available(self, cid, vtime):
                    return all(m.available(cid, vtime) for m in self.models)
            availability_model = _Composite(models)

    return availability_model, timeout_model
