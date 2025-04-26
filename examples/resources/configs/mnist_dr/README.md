# Running Experiments with Data Readiness (DR) Agents

This directory contains **Data Readiness (DR) agents** designed to identify and handle various data quality issues in the **MNIST dataset**.

## DR Agents Overview

Inside the `dr_agent/` directory, we provide five different custom DR agents. Each is responsible for detecting and optionally remedying a specific data readiness issue such as:

- Noise handling
- Class imbalance
- Duplicate samples
- Outlier management
- Memory usage optimization
- (Others depending on your implementation)

To run a specific DR agent, configure it in the server configuration YAML file.

## Example: Handling Class Imbalance

To handle **class imbalance (ci)**, update the `dragent_configs` in `server_fedavg_dragent.yaml` as shown:

```yaml
dragent_configs:
    dragent_path: ./resources/configs/mnist_dr/dr_agent/handle_ci.py
    dragent_name: DRAgentCI # Optional, if not specified, the last class will be used
    remedy_action: false  # Set to true to balance classes
```

The `dragent_name` is the name of the class defined in the `dragent_path` file. It is optional, and if not specified, the last Python class defined within the dragent file will be used.
Setting `remedy_action: true` will apply the fix (e.g., undersampling the majority class). The agent will then perform:

- Evaluation of the custom data readiness status
- Remedial action (if enabled)

## Example: Handling Duplicates
To detect or fix **duplicate entries**, modify the YAML config like so:

```yaml
dragent_configs:
    dragent_path: ./resources/configs/mnist_dr/dr_agent/handle_duplicates.py
    dragent_name: DRAgentDuplicates # Optional, if not specified, the last class will be used
    remedy_action: false  # Set to true to remove duplicates
```

Each DR agent defines its own metrics, rules and remedy mechanism.

## Running Experiments with MPI
To launch an experiment with 2 clients and 1 server using MPI, run the following while inside the `examples` directory:

```bash
mpiexec -n 3 python ./mpi/run_mpi.py \
    --server_config ./resources/configs/mnist_dr/server_fedavg_dragent.yaml \
    --client_config ./resources/configs/mnist_dr/client_1_dragent.yaml
```

## DR Reports
After running, DR evaluation reports are saved in the `output/` directory.

Each report includes:

- General DR metrics and visualizations as enabled under `data_readiness_configs` in `server_fedavg_dragent.yaml`
- Custom metrics specific to the selected DR agent

## Simulate data issues
Fortunately and unfortunately, most data issues are not naturally presented in the MNIST dataset as it is a very well-curated dataset, therefore, to evaluate DR agents effectively, we intentionally inject issues (e.g., duplicates, noise) to create a "noisy" and "non-AI-ready" version of MNIST dataset at `./resources/dataset/mnist_dataset_dr.py`.

## Notes
- You can extend the `BaseAgent` from `appfl.misc.data_readiness` and create new custom DR agents to handle any data related issues easily
