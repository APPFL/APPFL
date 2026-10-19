import math

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

pl = pytest.importorskip("lightning.pytorch")

from appfl.algorithm.trainer import LightningTrainer


class TinyModule(pl.LightningModule):
    def __init__(self, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", F.cross_entropy(self(x), y), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


class _StubLogger:
    def __init__(self):
        self.rows = []

    def log_title(self, titles):
        self.titles = titles

    def log_content(self, contents):
        self.rows.append(contents)

    def info(self, *args, **kwargs):
        pass


def _make_trainer(train_configs, client_id="Client1"):
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = (x[:, 0] > 0).long()
    dataset = TensorDataset(x, y)
    return LightningTrainer(
        model=TinyModule(),
        train_dataset=dataset,
        val_dataset=dataset,
        train_configs=OmegaConf.create(train_configs),
        logger=_StubLogger(),
        client_id=client_id,
    )


def test_epoch_mode_trains_and_returns_metrics():
    trainer = _make_trainer(
        {"mode": "epoch", "num_local_epochs": 2, "do_validation": True}
    )
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}

    for expected_round in (1, 2):
        trainer.train()
        params, metrics = trainer.get_parameters()
        assert metrics["round"] == expected_round
        assert "train_loss" in metrics and "val_loss" in metrics

    assert any(not torch.equal(before[k], params[k]) for k in before)
    assert all(v.device.type == "cpu" for v in params.values())
    assert trainer.logger.titles == ["Round", "Time", "train_loss", "val_loss"]
    assert len(trainer.logger.rows) == 2


def test_pre_validation_evaluates_received_model():
    trainer = _make_trainer(
        {"mode": "epoch", "num_local_epochs": 1, "do_pre_validation": True}
    )
    zeros = {k: torch.zeros_like(v) for k, v in trainer.model.state_dict().items()}
    trainer.load_parameters(zeros)
    trainer.train()
    _, metrics = trainer.get_parameters()
    # An all-zero linear model outputs equal logits for both classes: loss ln(2)
    assert metrics["pre_val_loss"] == pytest.approx(math.log(2))
    assert "val_loss" not in metrics  # do_validation defaults to False
    assert trainer.logger.titles == ["Round", "Time", "pre_val_loss", "train_loss"]


@pytest.mark.parametrize("option", ["send_gradient", "use_secure_agg", "use_dp"])
def test_unsupported_options_are_rejected(option):
    with pytest.raises(AssertionError, match=option):
        _make_trainer({"mode": "epoch", "num_local_epochs": 1, option: True})


def test_step_mode_and_load_parameters_round_trip():
    trainer = _make_trainer({"mode": "step", "num_local_steps": 3})
    trainer.train()
    params, metrics = trainer.get_parameters()
    assert "val_loss" not in metrics  # do_validation defaults to False

    zeros = {k: torch.zeros_like(v) for k, v in params.items()}
    trainer.load_parameters(zeros)
    assert all(torch.count_nonzero(v) == 0 for v in trainer.model.state_dict().values())


def test_mlflow_logs_all_rounds_into_one_run(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    trainer = _make_trainer(
        {
            "mode": "epoch",
            "num_local_epochs": 1,
            "train_batch_size": 16,
            "do_validation": True,
            "do_pre_validation": True,
            "mlflow_configs": {
                "enable_mlflow": True,
                "tracking_uri": tracking_uri,
                "experiment_name": "appfl-test",
            },
        }
    )
    for _ in range(3):
        trainer.train()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("appfl-test")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.run_name == "Client1"

    for name in ("train_loss", "pre_val_loss"):
        history = client.get_metric_history(runs[0].info.run_id, name)
        steps = [m.step for m in history]
        assert len(steps) == 3
        assert steps == sorted(steps) and len(set(steps)) == 3
    # Pre-validation goes to pre_val_loss only, so val_loss has one point per round
    assert len(client.get_metric_history(runs[0].info.run_id, "val_loss")) == 3


def test_client_agent_copies_client_mlflow_configs_to_trainer():
    from appfl.agent import ClientAgent

    agent = ClientAgent.__new__(ClientAgent)
    agent.client_agent_config = OmegaConf.create(
        {
            "client_id": "Client7",
            "train_configs": {
                "trainer": "LightningTrainer",
                "mode": "step",
                "num_local_steps": 1,
            },
            "mlflow_configs": {"enable_mlflow": False, "run_name": "site-a"},
        }
    )
    agent.model = TinyModule()
    agent.loss_fn = agent.metric = None
    agent.train_dataset = agent.val_dataset = TensorDataset(
        torch.randn(4, 4), torch.zeros(4, dtype=torch.long)
    )
    agent.logger = _StubLogger()
    agent._load_trainer()
    assert agent.trainer.train_configs.mlflow_configs.run_name == "site-a"
