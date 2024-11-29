import os
import json
import pprint
import argparse
import warnings
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.service.utils import APPFLxDataExchanger
from appfl.comm.globus_compute import GlobusComputeServerCommunicator

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

argparser = argparse.ArgumentParser()
argparser.add_argument("--run_aidr_only", action="store_true")
argparser.add_argument("--base_dir", type=str, required=True)
args = argparser.parse_args()

data_exchanger = APPFLxDataExchanger(base_dir=args.base_dir)
server_agent_config, client_agent_configs = data_exchanger.download_configurations(
    args.run_aidr_only
)

# Create server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Create server communicator
server_communicator = GlobusComputeServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_configs,
    logger=server_agent.logger,
    compute_token=server_agent.server_agent_config.appflx_configs.compute_token,
    openid_token=server_agent.server_agent_config.appflx_configs.openid_token,
)

if args.run_aidr_only:
    readiness_reports_dict = {}
    server_communicator.send_task_to_all_clients(task_name="data_readiness_report")
    readiness_reports = server_communicator.recv_result_from_all_clients()[1]
    # Restructure the data to match the function's expected input
    restructured_report = {}
    for client_endpoint_id, client_report in readiness_reports.items():
        # Assuming 'data_readiness' is the key containing the actual report
        client_data = client_report.get("data_readiness", {})
        for attribute, value in client_data.items():
            if attribute not in restructured_report:
                restructured_report[attribute] = {}
            restructured_report[attribute][client_endpoint_id] = value
    # Handle 'plots' separately if they exist
    if "plots" in restructured_report:
        plot_data = restructured_report.pop("plots")
        restructured_report["plots"] = {
            client_id: plot_data.get(client_id, {})
            for client_id in readiness_reports.keys()
        }
    # Call the data_readiness_report function
    server_agent.data_readiness_report(restructured_report)
    # Upload the reports and logs
    log_file = server_agent.logger.get_log_filepath()
    report_file = os.path.join(
        server_agent_config.client_configs.data_readiness_configs.output_dirname,
        f"{server_agent_config.client_configs.data_readiness_configs.output_filename}.html",
    )
    data_exchanger.upload_results(
        {
            "log.txt": log_file,
            "report.html": report_file,
        }
    )

else:
    # Get sample size from clients
    server_communicator.send_task_to_all_clients(task_name="get_sample_size")
    sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
    for client_endpoint_id, sample_size in sample_size_ret.items():
        server_agent.set_sample_size(client_endpoint_id, sample_size["sample_size"])

    # Train the model
    server_communicator.send_task_to_all_clients(
        task_name="train",
        model=server_agent.get_parameters(globus_compute_run=True),
        need_model_response=True,
    )

    model_futures = {}
    client_rounds = {}
    training_metadata = {}
    while not server_agent.training_finished():
        client_endpoint_id, client_model, client_metadata = (
            server_communicator.recv_result_from_one_client()
        )
        server_agent.logger.info(
            f"Received the following meta data from {client_endpoint_id}:\n{pprint.pformat(client_metadata)}"
        )
        if client_endpoint_id not in training_metadata:
            training_metadata[client_endpoint_id] = []
        training_metadata[client_endpoint_id].append(client_metadata)
        global_model = server_agent.global_update(
            client_endpoint_id,
            client_model,
            **client_metadata,
        )
        if isinstance(global_model, Future):
            model_futures[client_endpoint_id] = global_model
        else:
            if isinstance(global_model, tuple):
                global_model, metadata = global_model
            else:
                metadata = {}
            if client_endpoint_id not in client_rounds:
                client_rounds[client_endpoint_id] = 0
            client_rounds[client_endpoint_id] += 1
            metadata["round"] = client_rounds[client_endpoint_id]
            if not server_agent.training_finished():
                server_communicator.send_task_to_one_client(
                    client_endpoint_id,
                    task_name="train",
                    model=global_model,
                    metadata=metadata,
                    need_model_response=True,
                )
        # Deal with the model futures
        del_keys = []
        for client_endpoint_id in model_futures:
            if model_futures[client_endpoint_id].done():
                global_model = model_futures[client_endpoint_id].result()
                if isinstance(global_model, tuple):
                    global_model, metadata = global_model
                else:
                    metadata = {}
                if client_endpoint_id not in client_rounds:
                    client_rounds[client_endpoint_id] = 0
                client_rounds[client_endpoint_id] += 1
                metadata["round"] = client_rounds[client_endpoint_id]
                if not server_agent.training_finished():
                    server_communicator.send_task_to_one_client(
                        client_endpoint_id,
                        task_name="train",
                        model=global_model,
                        metadata=metadata,
                        need_model_response=True,
                    )
                del_keys.append(client_endpoint_id)
        for key in del_keys:
            model_futures.pop(key)
    # Upload the reports and logs
    log_file = server_agent.logger.get_log_filepath()
    # save the training metadata into a json file
    result_file = os.path.join(
        server_agent_config.server_configs.logging_output_dirname,
        "training_metadata.json",
    )
    with open(result_file, "w") as f:
        json.dump(training_metadata, f)
    data_exchanger.upload_results(
        {
            "log.txt": log_file,
            "training_metadata.json": result_file,
        }
    )

server_communicator.cancel_all_tasks()
server_communicator.shutdown_all_clients()
