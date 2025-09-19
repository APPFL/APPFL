import torch


def sd_from_tuple(x):
    return x[0] if isinstance(x, tuple) else x


def state_sub(a: dict, b: dict) -> dict:
    return {k: a[k].detach().cpu() - b[k].detach().cpu() for k in a.keys()}


def state_add(a: dict, b: dict, alpha: float = 1.0) -> dict:
    return {k: a[k].detach().cpu() + alpha * b[k].detach().cpu() for k in a.keys()}


def state_scale(a: dict, s: float) -> dict:
    return {k: v.detach().cpu() * s for k, v in a.items()}


def zeros_like_state(sd: dict) -> dict:
    return {k: v.detach().cpu().clone().zero_() for k, v in sd.items()}


def sd_l2norm(d: dict) -> float:
    return float(torch.sqrt(sum((v.float().view(-1) ** 2).sum() for v in d.values())))


def set_server_params(server, new_state: dict):
    """Robustly set server globals across APPFL variants."""
    if hasattr(server, "set_parameters"):
        try:
            server.set_parameters(new_state)
            return
        except TypeError:
            server.set_parameters((new_state,))
            return
        except Exception:
            pass
    if hasattr(server, "load_parameters"):
        try:
            server.load_parameters(new_state)
            return
        except Exception:
            pass
    for attr in ("model", "_model", "global_model"):
        m = getattr(server, attr, None)
        if m is not None and hasattr(m, "load_state_dict"):
            cpu_state = {k: v.detach().cpu() for k, v in new_state.items()}
            m.load_state_dict(cpu_state, strict=True)
            return
    if hasattr(server, "_set_parameters"):
        try:
            server._set_parameters(new_state)
            return
        except Exception:
            pass
    raise RuntimeError("Cannot set server parameters with this APPFL build.")
