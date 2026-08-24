## Federated Learning with AI Data Readiness Inspection on MNIST at NERSC

This tutorial demonstrates how to run a federated learning (FL) experiment on the MNIST dataset with APPFL at NERSC, **combined with an AI data readiness (DR) inspection step** that runs before training.

In federated learning the server never sees client data, so a silent data problem at one site (severe class imbalance, duplicates, noise, outliers) only shows up as a mysteriously bad global model. APPFL's data readiness support — which follows the [AI Data Readiness Inspector (AIDRIN)](https://dl.acm.org/doi/pdf/10.1145/3676288.3676296) methodology — has every client quantify the quality of its *own* dataset locally and report only aggregate metrics and plots to the server, which merges them into a single HTML dashboard.

On top of reporting, **CADRE (Customizable Assurance of Data REadiness) modules** let a client actually repair the issue before training. This tutorial uses `CADREModuleCI`, which detects class imbalance and rebalances the local dataset by undersampling the majority class.

### Notebooks

The tutorial consists of three Jupyter notebooks — one for the server and two for the clients, each client notebook simulating a separate data-owning site:

- `APPFL_Server_MNIST_AIDRIN.ipynb` — creates the FL server, decides which readiness metrics and which CADRE module all clients should use, serves the federation, and renders the merged readiness report.
- `APPFL_Client1_MNIST_AIDRIN.ipynb`, `APPFL_Client2_MNIST_AIDRIN.ipynb` — create the FL clients, inspect and repair their local data, send their readiness report, and run local training.

### How to run

1. Open **`APPFL_Server_MNIST_AIDRIN.ipynb`** and run it through the "Launch the server" cell. That cell prints an address such as `172.31.79.131:50051` and then blocks, serving the federation.
2. Open **both** client notebooks. In each, paste the address from step 1 into
   `client_agent_config.comm_configs.grpc_configs["server_uri"]`, then run all cells.
3. Both clients must be running: the readiness-report step and the synchronous training rounds block until every client reaches them.
4. When training finishes, go back to the server notebook and run the last section to render the merged data readiness report inline. The report is also written to `data_readiness_report*.html` in this directory.

> All three notebooks `os.chdir("../..")` in their first cell, so every relative path in this tutorial is resolved from the `examples/` directory. Run that cell only once per kernel.

### Running the clients as scripts

`run_client.py` is a script version of the client notebooks, for when a terminal is
easier than a second and third notebook kernel. Launch the server from the notebook
as above, then from this directory:

```bash
python run_client.py --client_id 1 --server_uri 172.31.79.131:50051
python run_client.py --client_id 2 --server_uri 172.31.79.131:50051
```

It runs the same steps as the notebooks — inspect, remedy, report, train — but
prints the metrics rather than rendering them. Pass `--save_plots` to write the
readiness plots here as PNG files, and `--use_ssl` for a hosted endpoint such as
`*.trac.appflx.link`. The script resolves its own paths, so it works from any
working directory. The merged HTML report is still written by the server notebook.

### Configuration

The notebooks reuse the shared configuration files under `examples/resources/configs/mnist_dr/`:

- `server_fedavg_cadremodule.yaml` — server, training, model and `data_readiness_configs`
- `client_1_cadremodule.yaml`, `client_2_cadremodule.yaml` — per-client dataset and connection settings
- `cadre_module/` — the five example CADRE modules (`handle_ci.py`, `handle_duplicates.py`, `handle_noise.py`, `handle_outliers.py`, `handle_mem.py`)

The server notebook overrides a few readiness settings *in memory* rather than editing the YAML on disk, so the same files keep working for the MPI and gRPC script examples described in `examples/resources/configs/mnist_dr/README.md`. One of those overrides is `output_dirname`, which points the merged report at this tutorial directory instead of `examples/output/`.

To try a different data issue, change `cadremodule_path` / `cadremodule_name` in the server notebook and restart all three notebooks. To report without repairing, set `remedy_action = False`.

Turning on more metrics makes the readiness report bigger, and some of them — `combine.feature_space_distribution` in particular — ship the full PCA projection of every sample to the server. If a client fails with a gRPC *message larger than max size* error, raise `max_message_size` in **both** the server and client configs (it is `10485760`, i.e. 10 MB, throughout this tutorial) rather than turning the metric back off. `run_client.py` takes `--max_message_size` for the same purpose.

### Writing your own CADRE module

Subclass `appfl.misc.data_readiness.BaseCADREModule` and implement three methods:

- `metric()` — compute a custom readiness number for the local dataset,
- `rule(metric_result)` — decide whether that number is bad enough to act on,
- `remedy(metric_result, logger)` — return `{"ai_ready_dataset": ..., "metadata": ...}` with the repaired dataset.

Then point `cadremodule_path` and `cadremodule_name` at your file.
