# [Prototype for LANL ONLY] PyTorch Lightning with APPFL

`LightningTrainer` lets an APPFL client train a PyTorch Lightning `LightningModule`. Each round of local training is one call to `pl.Trainer.fit`, so the module's `training_step`, `validation_step` and `configure_optimizers` are used as is, and the server aggregates the module's weights like any other PyTorch model.

## Installation

```bash
# Install from source only as this is not yet part of the APPFL release.
pip install -e ".[lightning,examples]" # or pip install -e ".[lightning]" if you already have the examples installed
```

The `lightning` extra installs `lightning`, and `mlflow` for optional logging. Add the `mpi` extra to run the MPI simulation.

## Files Added for Running Lightning Examples

- `examples/resources/model/lightning_cnn.py`: the MNIST CNN as a `LightningModule`. The loss, optimizer and logged metrics are defined here.
- `examples/resources/configs/mnist/server_fedavg_lightning.yaml`: FedAvg server config that selects `trainer: "LightningTrainer"` and points `model_configs` at `LightningCNN`. It works with any MNIST client config below.
- `examples/resources/configs/mnist/client_1_mlflow.yaml`, `client_2_mlflow.yaml`: the regular `client_1.yaml` / `client_2.yaml` with MLflow logging enabled. Use the regular ones to run without MLflow.

## Running with gRPC

Run each command in its own terminal, from the `examples` directory:

```bash
cd examples
python grpc/run_server.py --config ./resources/configs/mnist/server_fedavg_lightning.yaml
python grpc/run_client.py --config ./resources/configs/mnist/client_1_mlflow.yaml
python grpc/run_client.py --config ./resources/configs/mnist/client_2_mlflow.yaml
```

The server sends the trainer settings and the module source to the clients, so the client configs only describe the local data, the server address, the device and, optionally, MLflow. Every client needs `lightning` installed.

## Running as a serial or MPI simulation

```bash
cd examples
# Serial: the server and all clients in one process
python serial/run_serial.py \
  --server_config ./resources/configs/mnist/server_fedavg_lightning.yaml \
  --client_config ./resources/configs/mnist/client_1_mlflow.yaml \
  --num_clients 2
# MPI: rank 0 is the server and every other rank is a client, so -n is the number of clients + 1
mpiexec -n 3 python mpi/run_mpi.py \
  --server_config ./resources/configs/mnist/server_fedavg_lightning.yaml \
  --client_config ./resources/configs/mnist/client_1_mlflow.yaml
```

Both scripts load the one `--client_config` for every client and then set each client's id (`Client1`, `Client2`, ...) and data partition, so each client still gets its own MLflow run. Use `client_1.yaml` instead to run without MLflow.

## Training settings

These keys live under `client_configs.train_configs` in the server config that are shared across all clients, except `device`, which is set per client in the client config:

| Key | Meaning |
| --- | --- |
| `mode` | `"epoch"` (uses `num_local_epochs`) or `"step"` (uses `num_local_steps`) per round |
| `device` | `"cpu"`, `"cuda"`, `"cuda:<index>"` or `"mps"` |
| `train_batch_size`, `val_batch_size`, `train_data_shuffle`, `num_workers` | Data loader settings |
| `do_validation` | Run the module's `validation_step` at the end of each local epoch |
| `do_pre_validation` | Run the module's `validation_step` on the received global model before local training; its metrics are reported with a `pre_` prefix (e.g. `pre_val_accuracy`) |

`optim`, `optim_args` and `loss_fn` are no longer used for training (compared to the pytorch-compatible `VanillaTrainer`), since the module defines them. `loss_fn_path` and `metric_path` are only used if the server runs its own validation.

Whatever the module logs with `self.log` inside the `training_step` or `validation_step` is printed in the client log each round and sent to the server with the model.

## MLflow logging

MLflow settings are client specific, like `wandb_configs`: each client can point to its own tracking server and use its own run name. Add `mlflow_configs` at the top level of the **client** config, as in `client_1_mlflow.yaml`:

```yaml
mlflow_configs:
  enable_mlflow: True
  experiment_name: "appfl-lightning-mnist"
  tracking_uri: "./mlruns"     # or a tracking server, e.g. http://localhost:5001
  # run_name: "Client1"     # defaults to the client id
```

Each client gets one MLflow run that covers all rounds. It holds the module's hyperparameters (from `save_hyperparameters()`), every metric logged with `self.log`, the `pre_*` metrics when `do_pre_validation` is on, and a `round` metric.

With the default `tracking_uri: "./mlruns"`, the runs are written to `examples/mlruns`. To view them:

```bash
cd examples
mlflow ui --backend-store-uri ./mlruns
```

To log to an MLflow tracking server instead, start one and point `tracking_uri` at it:

```bash
mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db
```

```yaml
  tracking_uri: "http://localhost:5001"
```

Then open http://localhost:5001.

## Using your own LightningModule

1. Write the module with `training_step`, `configure_optimizers` and, optionally, `validation_step`. Import Lightning as `lightning.pytorch`. The trainer uses `lightning.pytorch` whenever the `lightning` package is installed, and a module built on the standalone `pytorch_lightning` package is then rejected with a clear error.
2. Make sure `forward()` works on its own, because the server calls it for optional server-side validation.
3. Point `model_configs.model_path`, `model_name` and `model_kwargs` in the server config at your module, and the client `data_configs` at your dataset.

## Current limitations

`LightningTrainer` is a prototype that covers the basic federated training loop. If you need any of the features below, let us know and we can add them together.

- One device per client. Multi-GPU (DDP) inside a client is not supported yet.
- The optimizer state is rebuilt every round, like the default `VanillaTrainer`.
- `LightningTrainer` always sends the full model weights, so it can't be used with server algorithms that need clients to send updates (`send_gradient: True`, e.g. FedAsync, FedBuff, FedCompass). Secure aggregation (`use_secure_agg`) and differential privacy (`use_dp`) are not supported yet as well. The trainer stops with an error if any of these three options is set to `True`.
