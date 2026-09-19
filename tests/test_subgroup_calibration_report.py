"""CADRE loading, report output, and gRPC error handling."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator
from appfl.comm.grpc.grpc_communicator_pb2 import ClientHeader, CustomActionRequest
from appfl.comm.grpc.grpc_communicator_pb2_grpc import (
    GRPCCommunicatorStub,
    add_GRPCCommunicatorServicer_to_server,
)
from appfl.comm.grpc.utils import proto_to_databuffer
from appfl.misc.data_readiness import SubgroupCalibrationCADREModule
from appfl.misc.data_readiness.calibration import build_calibration_report
from appfl.misc.data_readiness.report import (
    generate_html_content,
    get_calibration_summary,
)


def _close_logger(agent):
    for handler in list(agent.logger.logger.handlers):
        handler.close()
        agent.logger.logger.removeHandler(handler)


@pytest.fixture
def server_agent(tmp_path):
    agent = ServerAgent(
        OmegaConf.create(
            {
                "server_configs": {
                    "num_clients": 2,
                    "logging_output_dirname": str(tmp_path),
                },
                "client_configs": {
                    "data_readiness_configs": {"output_dirname": str(tmp_path)}
                },
            }
        )
    )
    yield agent
    _close_logger(agent)


def _payload(group="A", k=5):
    return build_calibration_report(
        [0.05] * 3 + [0.15] * 3,
        [0, 0, 1, 0, 1, 0],
        [group] * 6,
        min_cell_count=k,
        release_n_bins=5,
        min_outcome_count=1,
    )


def _readiness(payload):
    return {"specified_metrics": {"first": {"subgroup_calibration": payload}}}


def test_cadre_loader_and_remedy_follow_appfl_signature(tmp_path):
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "examples/data_readiness/subgroup_calibration/cadre_module.py"
    )
    config = OmegaConf.create(
        {
            "client_id": "site",
            "train_configs": {"logging_output_dirname": str(tmp_path)},
            "data_readiness_configs": {
                "dr_metrics": {
                    "cadremodule_configs": {
                        "cadremodule_path": str(wrapper),
                        "cadremodule_name": "SubgroupCalibrationCADREModule",
                        "cadremodule_kwargs": {
                            "release_n_bins": 5,
                            "min_cell_count": 5,
                        },
                    }
                }
            },
        }
    )
    client = ClientAgent(config)
    dataset = SimpleNamespace(
        predictions=[0.05] * 3 + [0.15] * 3,
        outcomes=[0] * 6,
        groups=["A"] * 6,
        data_input=torch.zeros((6, 1)),
        data_label=torch.zeros(6),
    )
    try:
        client.train_dataset = dataset
        report = client.generate_readiness_report(config)
        assert set(report["specified_metrics"]) == {"subgroup_calibration"}
        payload = report["specified_metrics"]["subgroup_calibration"]
        assert payload["n_bins"] == 5
        assert payload["cells"]["A"]["0"]["n"] == 6
        assert "client_id" not in payload
        assert client.adapt_data(config) is None
        assert client.train_dataset is dataset
    finally:
        _close_logger(client)


def test_dataset_requires_stored_predictions():
    with pytest.raises(ValueError, match="predictions"):
        SubgroupCalibrationCADREModule([]).metric()


def test_report_combines_clients_and_escapes_group_labels(server_agent, tmp_path):
    label = '<script>alert("x")</script>患者'
    readiness = _readiness(_payload(label))
    readiness["specified_metrics"]["second"] = {"subgroup_calibration": _payload(label)}
    server_agent.data_readiness_report(readiness)
    result = json.loads(
        (tmp_path / "data_readiness_report_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["groups"][label]["n"] == 12
    assert result["scope"] == "retained_patients"
    html = (tmp_path / "data_readiness_report.html").read_text(encoding="utf-8")
    assert "Subgroup calibration" in html
    assert "retained patients only" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "sum_sq_err" not in html
    assert "患者" in html
    server_agent.data_readiness_report(readiness)
    assert (tmp_path / "data_readiness_report_calibration_1.json").exists()


def test_other_cadre_reports_keep_working(server_agent, tmp_path):
    readiness = {"specified_metrics": {"first": {"mean": 0.5}, "second": {"mean": 0.6}}}
    assert get_calibration_summary(readiness) is None
    server_agent.data_readiness_report(readiness)
    assert (tmp_path / "data_readiness_report.html").exists()
    assert not list(tmp_path.glob("*_calibration*.json"))
    assert "Subgroup calibration" not in generate_html_content(readiness)


def test_all_suppressed_report_is_unavailable():
    result = get_calibration_summary(_readiness(_payload(k=10)))
    assert result["status"] == "unavailable"
    html = generate_html_content(_readiness(_payload(k=10)))
    assert "Unavailable" in html
    assert "Full-cohort coverage is unknown" in html


@pytest.mark.parametrize("second", [{}, {"mean": 1}])
def test_missing_client_calibration_is_rejected(second):
    readiness = _readiness(_payload())
    readiness["specified_metrics"]["second"] = second
    with pytest.raises(ValueError, match="all clients"):
        get_calibration_summary(readiness)


def test_client_missing_from_specified_metrics_is_rejected():
    readiness = _readiness(_payload())
    readiness["plots"] = {"first": {}, "second": {}}
    with pytest.raises(ValueError, match="all clients"):
        get_calibration_summary(readiness)


def test_invalid_report_is_not_written(server_agent, tmp_path):
    readiness = _readiness(_payload())
    readiness["specified_metrics"]["second"] = {"subgroup_calibration": _payload(k=10)}
    with pytest.raises(ValueError, match="matching"):
        server_agent.data_readiness_report(readiness)
    assert not list(tmp_path.glob("data_readiness_report*"))


@pytest.fixture
def grpc_reports(server_agent):
    server_pool = ThreadPoolExecutor(max_workers=4)
    server = grpc.server(server_pool)
    communicator = GRPCServerCommunicator(server_agent, logger=server_agent.logger)
    add_GRPCCommunicatorServicer_to_server(communicator, server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = GRPCCommunicatorStub(channel)
    workers = ThreadPoolExecutor(max_workers=4)

    def request(client_id, metadata, timeout=5):
        request = CustomActionRequest(
            header=ClientHeader(client_id=client_id),
            action="get_data_readiness_report",
            meta_data=yaml.safe_dump(metadata),
        )
        return stub.InvokeCustomAction(proto_to_databuffer(request), timeout=timeout)

    try:
        yield communicator, request, workers
    finally:
        server.stop(0).wait(5)
        channel.close()
        workers.shutdown(wait=True)
        server_pool.shutdown(wait=True)


def _metadata(k=5):
    return {"specified_metrics": {"subgroup_calibration": _payload(k=k)}}


def _pending_report(communicator, client_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with communicator._dr_metrics_lock:
            future = communicator._dr_metrics_futures.get(client_id)
        if future is not None:
            return future
        time.sleep(0.01)
    pytest.fail("client did not submit its readiness report")


@pytest.mark.parametrize("second", [_metadata(k=10), {}, {"plots": {}}])
def test_grpc_releases_all_waiters_after_invalid_report_and_recovers(
    grpc_reports, tmp_path, second
):
    _, request, workers = grpc_reports
    failed = [
        workers.submit(list, request("first", _metadata())),
        workers.submit(list, request("second", second)),
    ]
    _, pending = wait(failed, timeout=8)
    assert not pending
    for future in failed:
        with pytest.raises(grpc.RpcError) as error:
            future.result()
        assert error.value.code() == grpc.StatusCode.INTERNAL
    assert not list(tmp_path.glob("data_readiness_report*"))
    healthy = [
        workers.submit(list, request(client, _metadata()))
        for client in ("first", "second")
    ]
    for future in healthy:
        assert future.result(timeout=8)
    result = json.loads(
        (tmp_path / "data_readiness_report_calibration.json").read_text()
    )
    assert result["groups"]["A"]["n"] == 12


def test_grpc_cancellation_releases_waiters_and_allows_another_batch(
    grpc_reports, server_agent, tmp_path
):
    communicator, request, workers = grpc_reports
    server_agent.num_clients = 3
    first = request("first", _metadata())
    first_result = workers.submit(list, first)
    first_pending = _pending_report(communicator, "first")
    second_result = workers.submit(list, request("second", _metadata()))
    second_pending = _pending_report(communicator, "second")
    first.cancel()
    for pending in (first_pending, second_pending):
        with pytest.raises(RuntimeError, match="cancelled"):
            pending.result(timeout=3)
    for result in (first_result, second_result):
        with pytest.raises(grpc.RpcError):
            result.result(timeout=3)
    assert not communicator._dr_metrics_futures
    assert not communicator._dr_metrics
    assert not list(tmp_path.glob("data_readiness_report*"))

    healthy = [workers.submit(list, request("first", _metadata()))]
    current = _pending_report(communicator, "first")
    communicator._cancel_readiness_report("first", first_pending)
    assert not current.done()
    healthy.extend(
        workers.submit(list, request(client, _metadata()))
        for client in ("second", "third")
    )
    for result in healthy:
        assert result.result(timeout=8)
    summary = json.loads(
        (tmp_path / "data_readiness_report_calibration.json").read_text()
    )
    assert summary["groups"]["A"]["n"] == 18


@pytest.mark.parametrize("early_callback", [False, True])
def test_grpc_cancellation_during_callback_registration(grpc_reports, early_callback):
    communicator, request, workers = grpc_reports
    first = workers.submit(list, request("first", _metadata()))
    _pending_report(communicator, "first")

    class CancelledContext:
        def add_callback(self, callback):
            if early_callback:
                callback()
            return early_callback

        def is_active(self):
            return True

        def set_code(self, code):
            pass

        def set_details(self, details):
            pass

    cancelled_request = CustomActionRequest(
        header=ClientHeader(client_id="second"),
        action="get_data_readiness_report",
        meta_data=yaml.safe_dump(_metadata()),
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        list(
            communicator.InvokeCustomAction(
                proto_to_databuffer(cancelled_request), CancelledContext()
            )
        )
    with pytest.raises(grpc.RpcError):
        first.result(timeout=3)
    assert not communicator._dr_metrics_futures
    assert not communicator._dr_metrics
    healthy = [
        workers.submit(list, request(client, _metadata()))
        for client in ("first", "second")
    ]
    for future in healthy:
        assert future.result(timeout=8)
