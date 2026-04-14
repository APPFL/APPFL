# Federated Genome-Wide Association Study (GWAS) using TES

This example demonstrates how to run a federated Genome-Wide Association Study (GWAS) using APPFL with the GA4GH Task Execution Service (TES) as the compute backend. 

## APPFL Installation

Install APPFL in a clean conda environment:

```bash
conda create -n appfl-env python=3.10 -y
conda activate appfl-env
git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
cd APPFL
pip install -e ".[examples]"
```

## How to Join the Federation

A template client configuration file [`clients-template.yaml`](./clients-template.yaml) is provided in this directory with ten pre-configured clients. Copy the template, then fill in `tes_endpoint: "<your_tes_endpoint>"` for each client you want to use. Remove entries for any clients you do not need, and update `server_configs.num_clients` in `server.yaml` to match.

> **💡 Note:** The sample data provided covers Site1 through Site5 only. Ensure the `site_id` in each client's `dataset_kwargs` is set to a value within that range. The existing ten entries in `clients-template.yaml` are already configured correctly. If you add more than ten clients, you must still keep each `site_id` within Site1–Site5.

## S3 Bucket for Data Transmission

The `server.yaml` file uses the S3 bucket `appfl-test-demo-bkt` to transfer model data between the server and clients. Only the server needs access to this bucket; clients do not.

If you do not have access to `appfl-test-demo-bkt`, create your own bucket and update the bucket name in `server.yaml`:

```bash
aws s3 mb s3://your-own-bucket-name
```

To verify your access to the bucket, run:

```bash
aws s3 ls s3://appfl-test-demo-bkt
```

A successful response lists the bucket contents (or an empty list). An error response means you do not have the required permissions.

## Run the Example with TES

> **💡 Note:** Update the `tes_endpoint` for each client in `clients-template.yaml` before running.

```bash
python tes/run.py --server_config ./resources/config_tes/gwas/server.yaml --client_config ./resources/config_tes/gwas/clients-template.yaml
```

## [Advanced] Run with Map Visualization

> **💡 Note:** Map visualization currently requires TES endpoints to be specified as IP addresses rather than hostnames.

### Install the Visualization Package

In the same conda environment, install `appfl-log`:

```bash
git clone https://github.com/APPFL/appfl-log.git
cd appfl-log
git checkout ga4gh/demo
pip install -e .
```

### Run with Visualization

```bash
python tes/run_viz.py --server_config ./resources/config_tes/gwas/server.yaml --client_config ./resources/config_tes/gwas/clients-template.yaml
```

The map visualization will be available at `http://localhost:7070` while the server is running. Once training finishes, you can replay the run at any time with:

```bash
hivewatch map run --runs-dir runs --port 7070
```