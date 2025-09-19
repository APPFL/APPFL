import os
import importlib.util
from omegaconf import OmegaConf, DictConfig
from appfl.agent import ServerAgent, ClientAgent

try:
    from custom_agents import CustomServerAgent as _CustomServerAgent
except Exception:
    _CustomServerAgent = None


def ensure_key(cfg_obj, key, default):
    if not hasattr(cfg_obj, key):
        setattr(cfg_obj, key, default)


def require_path(label: str, p: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"{label} not found: {p}")


def ensure_client_ready(c: ClientAgent):
    if getattr(c, "trainer", None) is not None:
        return
    if hasattr(c, "_prepare") and callable(c._prepare):
        c._prepare()
    elif hasattr(c, "prepare") and callable(c.prepare):
        c.prepare()
    if getattr(c, "trainer", None) is None:
        raise RuntimeError(
            "Client trainer not initialized. Check client YAML sections (trainer/model/loss/metric/data)."
        )


def fix_metric_symbol(path: str, name: str) -> str:
    spec = importlib.util.spec_from_file_location("metric_mod", path)
    if spec is None or spec.loader is None:
        return name
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return name
    for cand in (name, "ACC", "acc", "accuracy", "Accuracy"):
        if hasattr(mod, cand):
            return cand
    return name


def build_server(cfg: DictConfig) -> ServerAgent:
    s_cfg = OmegaConf.load(str(cfg.paths.server_config))
    s_cfg.server_configs.num_clients = int(cfg.system.num_clients)
    s_cfg.server_configs.num_global_epochs = int(cfg.system.num_rounds)
    if hasattr(s_cfg.server_configs, "scheduler_kwargs"):
        s_cfg.server_configs.scheduler_kwargs.num_clients = int(cfg.system.num_clients)
    if hasattr(s_cfg.server_configs, "aggregator_kwargs"):
        s_cfg.server_configs.aggregator_kwargs.num_clients = int(cfg.system.num_clients)
    if _CustomServerAgent is not None:
        return _CustomServerAgent(
            s_cfg, threshold=1, momentum_coeff=float(cfg.aggregation.momentum_coeff)
        )
    return ServerAgent(s_cfg)


def build_clients(cfg: DictConfig):
    clients = {}
    base = OmegaConf.load(str(cfg.paths.client_config))

    n = int(cfg.system.num_clients)
    gpus = [g.strip() for g in str(cfg.system.gpus).split(",") if g.strip()]
    assert len(gpus) >= n, f"Need {n} GPUs, got {gpus}"

    # resources
    res = getattr(cfg, "resources", OmegaConf.create({}))
    model_path = str(
        getattr(getattr(res, "model", {}), "path", "./resources/model/simple_cnn.py")
    )
    model_name = str(getattr(getattr(res, "model", {}), "name", "SimpleCNN"))
    loss_path = str(
        getattr(getattr(res, "loss", {}), "path", "./resources/loss/celoss.py")
    )
    loss_name = str(getattr(getattr(res, "loss", {}), "name", "CELoss"))
    metric_path = str(
        getattr(getattr(res, "metric", {}), "path", "./resources/metric/acc.py")
    )
    metric_name = str(getattr(getattr(res, "metric", {}), "name", "ACC"))
    data_path = str(
        getattr(
            getattr(res, "dataset", {}),
            "path",
            "./resources/dataset/custom_mnist_dataset.py",
        )
    )
    data_name = str(getattr(getattr(res, "dataset", {}), "name", "get_custom_mnist"))

    for p, lbl in [
        (model_path, "Model file"),
        (loss_path, "Loss file"),
        (metric_path, "Metric file"),
        (data_path, "Dataset file"),
    ]:
        require_path(lbl, p)

    warmup_steps = int(cfg.case2.warmup_steps)

    for i in range(n):
        cid = f"Client{i + 1}"
        c_cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        c_cfg.client_id = cid

        ensure_key(c_cfg, "train_configs", OmegaConf.create({}))
        ensure_key(c_cfg, "data_configs", OmegaConf.create({}))
        ensure_key(c_cfg, "model_configs", OmegaConf.create({}))
        ensure_key(c_cfg.data_configs, "dataset_kwargs", OmegaConf.create({}))

        tc = c_cfg.train_configs
        ensure_key(tc, "trainer", "VanillaTrainer")
        ensure_key(tc, "mode", "step")
        ensure_key(tc, "num_local_steps", warmup_steps)
        ensure_key(tc, "optim", "Adam")
        ensure_key(tc, "optim_args", OmegaConf.create({"lr": 0.001}))
        ensure_key(tc, "loss_fn_path", loss_path)
        ensure_key(tc, "loss_fn_name", loss_name)
        ensure_key(tc, "metric_path", metric_path)
        ensure_key(tc, "metric_name", metric_name)
        tc.metric_name = fix_metric_symbol(tc.metric_path, tc.metric_name)

        mc = c_cfg.model_configs
        ensure_key(mc, "model_path", model_path)
        ensure_key(mc, "model_name", model_name)

        dc = c_cfg.data_configs
        ensure_key(dc, "dataset_path", data_path)
        ensure_key(dc, "dataset_name", data_name)

        dkw = c_cfg.data_configs.dataset_kwargs
        dkw.num_clients = n
        dkw.client_id = i
        dkw.partition_strategy = str(
            getattr(getattr(cfg, "data", {}), "partition", "iid")
        )
        dkw.alpha = float(getattr(getattr(cfg, "data", {}), "alpha", 0.5))
        dkw.seed = int(cfg.system.seed)
        dkw.visualization = i == 0

        tc.device = gpus[i]
        c = ClientAgent(c_cfg)
        ensure_client_ready(c)
        clients[cid] = c

    return clients, [f"Client{i + 1}" for i in range(n)]
