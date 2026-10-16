"""Evaluate stored predictions through APPFL's CADRE report pathway."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import grpc
import numpy as np
import torch
from omegaconf import OmegaConf

from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.grpc import GRPCClientCommunicator, GRPCServerCommunicator
from appfl.comm.grpc.grpc_communicator_pb2_grpc import (
    add_GRPCCommunicatorServicer_to_server,
)


class CalibrationDataset(torch.utils.data.Dataset):
    """Stored binary predictions, outcomes, and subgroup labels."""

    def __init__(self, predictions, outcomes, groups):
        self.predictions = predictions
        self.outcomes = outcomes
        self.groups = groups
        self.data_input = torch.as_tensor(predictions).reshape(-1, 1)
        self.data_label = torch.as_tensor(outcomes)

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]


def make_sites(seed=42):
    """Four synthetic hospitals with different subgroup and outcome mixes."""
    rng = np.random.default_rng(seed)
    settings = (
        ("large_urban", 2000, 0.30, 0.70),
        ("midsize", 500, 0.15, 0.50),
        ("small_regional", 120, 0.45, 0.85),
        ("tiny_clinic", 40, 0.25, 0.40),
    )
    sites = {}
    for name, count, rate, group_share in settings:
        groups = rng.choice(["F", "M"], count, p=[group_share, 1 - group_share])
        logits = np.log(rate / (1 - rate)) + rng.normal(0, 1.2, count)
        probabilities = 1 / (1 + np.exp(-logits))
        outcomes = (rng.random(count) < probabilities).astype(int)
        predictions = probabilities.copy()
        if name == "midsize":
            mask = groups == "M"
            predictions[mask] = 1 / (1 + np.exp(-(0.5 + 1.8 * logits[mask])))
        sites[name] = CalibrationDataset(predictions, outcomes, groups)
    return sites


def _close_logger(agent):
    for handler in list(agent.logger.logger.handlers):
        handler.close()
        agent.logger.logger.removeHandler(handler)


def _client_report(client, config):
    client.load_config(config)
    return client.generate_readiness_report(config)


def _network_reports(server_agent, clients):
    server_pool = ThreadPoolExecutor(max_workers=8)
    server = grpc.server(server_pool)
    communicator = GRPCServerCommunicator(server_agent, logger=server_agent.logger)
    add_GRPCCommunicatorServicer_to_server(communicator, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if not port:
        server_pool.shutdown(wait=True)
        raise RuntimeError("could not bind the local gRPC server")
    server.start()
    client_pool = ThreadPoolExecutor(max_workers=len(clients))

    def submit(client, connection):
        config = connection.get_configuration()
        report = _client_report(client, config)
        connection.invoke_custom_action(action="get_data_readiness_report", **report)

    try:
        connections = [
            GRPCClientCommunicator(
                client_id=client.get_id(),
                server_uri=f"127.0.0.1:{port}",
                logger=client.logger,
                use_ssl=False,
            )
            for client in clients
        ]
        futures = [
            client_pool.submit(submit, client, connection)
            for client, connection in zip(clients, connections)
        ]
        _, pending = wait(futures, timeout=60)
        if pending:
            raise TimeoutError("the calibration clients did not finish within 60s")
        for future in futures:
            future.result()
    finally:
        server.stop(0).wait(5)
        client_pool.shutdown(wait=True)
        server_pool.shutdown(wait=True)


def run(output_dir, network=False, k=5, release_bins=5, min_outcomes=0):
    """Save a CADRE report and return its aggregated calibration summary."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = set(output.glob("data_readiness_report_calibration*.json"))
    config = OmegaConf.load(Path(__file__).with_name("server.yaml"))
    readiness = config.client_configs.data_readiness_configs
    readiness.output_dirname = str(output)
    module = readiness.dr_metrics.cadremodule_configs
    module.cadremodule_path = str(Path(__file__).with_name("cadre_module.py").resolve())
    module.cadremodule_kwargs.min_cell_count = k
    module.cadremodule_kwargs.release_n_bins = release_bins
    module.cadremodule_kwargs.min_outcome_count = min_outcomes
    config.server_configs.logging_output_dirname = str(output)
    server_agent = ServerAgent(config)
    clients = []
    try:
        for client_id, dataset in make_sites().items():
            client = ClientAgent(
                OmegaConf.create(
                    {
                        "client_id": client_id,
                        "train_configs": {"logging_output_dirname": str(output)},
                    }
                )
            )
            client.train_dataset = dataset
            clients.append(client)
        if network:
            _network_reports(server_agent, clients)
        else:
            reports = {}
            for client in clients:
                report = _client_report(client, server_agent.get_client_configs())
                for key, value in report.items():
                    reports.setdefault(key, {})[client.get_id()] = value
            server_agent.data_readiness_report(
                reports, expected_client_ids=[client.get_id() for client in clients]
            )
        created = set(output.glob("data_readiness_report_calibration*.json")) - previous
        if len(created) != 1:
            raise RuntimeError("expected one new calibration JSON report")
        result = json.loads(created.pop().read_text(encoding="utf-8"))
        print(json.dumps(result, indent=2))
        print(f"Reports: {output}")
        return result
    finally:
        for client in clients:
            _close_logger(client)
        _close_logger(server_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/subgroup_calibration")
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--release-bins", type=int, choices=(1, 2, 5, 10), default=5)
    parser.add_argument("--min-outcomes", type=int, default=0)
    args = parser.parse_args()
    if args.k < 1 or args.min_outcomes < 0:
        parser.error("k must be positive and min-outcomes must be non-negative")
    run(args.output_dir, args.network, args.k, args.release_bins, args.min_outcomes)
