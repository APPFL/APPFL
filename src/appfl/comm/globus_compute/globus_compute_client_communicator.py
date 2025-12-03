def globus_compute_client_entry_point(
    task_name="N/A",
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    """
    Entry point for the Globus Compute client endpoint for federated learning.
    :param `task_name`: The name of the task to be executed.
    :param `client_agent_config`: The configuration for the client agent.
    :param `model`: [Optional] The model to be used for the task.
    :param `meta_data`: [Optional] The metadata for the task.
    :return `model_local`: The local model after the task is executed. [Return `None` if the task does not return a model.]
    :return `meta_data_local`: The local metadata after the task is executed. [Return `{}` if the task does not return metadata.]
    """
    # The new way to execute task for globus compute is to import the executor functions
    # from client's local appfl instead of sending the function logics within the function.
    try:
        from appfl.comm.utils.executor import (
            get_sample_size_executor,
            data_readiness_report_executor,
            train_executor,
        )

        if task_name == "get_sample_size":
            return get_sample_size_executor(client_agent_config=client_agent_config)
        elif task_name == "data_readiness_report":
            return data_readiness_report_executor(
                client_agent_config=client_agent_config
            )
        elif task_name == "train":
            return train_executor(
                client_agent_config=client_agent_config,
                model=model,
                meta_data=meta_data,
            )
        else:
            raise NotImplementedError(
                f"Task {task_name} is not implemented in the client endpoint."
            )
    # Continue to support the old client appfl version until version 2.0.0
    except (ModuleNotFoundError, ImportError):
        from appfl.agent import ClientAgent
        from appfl.comm.utils.client_utils import (
            load_global_model,
            send_local_model,
        )

        client_agent = ClientAgent(client_agent_config=client_agent_config)

        if (
            hasattr(client_agent_config, "data_readiness_configs")
            and hasattr(
                client_agent_config.data_readiness_configs.dr_metrics,
                "cadremodule_configs",
            )
            and hasattr(
                client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs,
                "remedy_action",
            )
            and client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
        ):
            client_agent.adapt_data(client_config=client_agent_config)

        if model is not None:
            model = load_global_model(client_agent.client_agent_config, model)
            client_agent.load_parameters(model)

        if task_name == "get_sample_size":
            return None, {
                "sample_size": client_agent.get_sample_size(),
                "_deprecated": True,
            }

        elif task_name == "data_readiness_report":
            return None, {
                "data_readiness": client_agent.generate_readiness_report(
                    client_agent_config
                ),
                "_deprecated": True,
            }

        elif task_name == "train":
            client_agent.train(**meta_data)
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, meta_data_local = local_model
            else:
                meta_data_local = {}
            local_model = send_local_model(
                client_agent.client_agent_config,
                local_model,
                meta_data["local_model_key"]
                if "local_model_key" in meta_data
                else None,
                meta_data["local_model_url"]
                if "local_model_url" in meta_data
                else None,
            )
            meta_data_local["_deprecated"] = True
            return local_model, meta_data_local
        else:
            raise NotImplementedError(
                f"Task {task_name} is not implemented in the client endpoint."
            )


def globus_compute_client_entry_point_ds(
    task_name="N/A",
    client_agent_config_file=None,
    model_file=None,
    meta_data_file=None,
):
    """
    Entry point for the Globus Compute client endpoint for federated learning.
    :param `task_name`: The name of the task to be executed.
    :param `client_agent_config`: The configuration for the client agent.
    :param `model`: [Optional] The model to be used for the task.
    :param `meta_data`: [Optional] The metadata for the task.
    :return `model_local`: The local model after the task is executed. [Return `None` if the task does not return a model.]
    :return `meta_data_local`: The local metadata after the task is executed. [Return `{}` if the task does not return metadata.]
    """
    # The new way to execute task for globus compute is to import the executor functions
    # from client's local appfl instead of sending the function logics within the function.
    import json
    import pickle
    import time
    from omegaconf import OmegaConf
    from globus_compute_sdk import Executor, Client
    from globus_compute_sdk.serialize import CombinedCode
    from appfl.comm.utils.executor import (
        get_sample_size_executor_ds,
        train_executor_ds,
    )
    from appfl.comm.globus_compute.globus_compute_server_communicator import (
        data_transfer,
    )

    # load in from local files
    client_agent_config = OmegaConf.load(client_agent_config_file)

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    with open(meta_data_file) as f:
        meta_data = json.load(f)

    gcc = Client(
        code_serialization_strategy=CombinedCode(),
    )
    gce = Executor(client=gcc)
    gce.endpoint_id = client_agent_config.server_endpoint_id

    data_transfer_uuid = gcc.register_function(data_transfer)

    if task_name == "get_sample_size_ds":
        model, meta_data = get_sample_size_executor_ds(
            client_agent_config=client_agent_config
        )
        if meta_data != None:
            task_id = gcc.run(
                model,
                meta_data,
                client_agent_config.client_id,
                endpoint_id=gce.endpoint_id,
                function_id=data_transfer_uuid,
            )
            while True:
                status = gcc.get_batch_result([task_id])[task_id]

                if not status["pending"]:
                    if status["status"] == "success":
                        break
                else:
                    time.sleep(5)

    elif task_name == "train_ds":
        model, meta_data = train_executor_ds(
            client_agent_config=client_agent_config,
            model=model,
            meta_data=meta_data,
        )
        if model != None:
            task_id = gcc.run(
                model,
                meta_data,
                client_agent_config.client_id,
                endpoint_id=gce.endpoint_id,
                function_id=data_transfer_uuid,
            )
            while True:
                status = gcc.get_batch_result([task_id])[task_id]

                if not status["pending"]:
                    if status["status"] == "success":
                        break
                else:
                    time.sleep(5)
    else:
        raise NotImplementedError(
            f"Task {task_name} is not implemented in the client endpoint."
        )


def data_transfer(client_agent_config, model, meta_data):
    import os
    import json
    import pickle
    from omegaconf import OmegaConf

    home = os.environ.get("HOME")

    client_config_path = os.path.join(home, "client_config.yaml")
    with open(client_config_path, "w") as f:
        OmegaConf.save(client_agent_config, f)
        f.flush()
        os.fsync(f.fileno())

    model_path = os.path.join(home, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        f.flush()
        os.fsync(f.fileno())

    meta_data_path = os.path.join(home, "meta_data.json")
    with open(meta_data_path, "w") as f:
        json.dump(meta_data, f)
        f.flush()
        os.fsync(f.fileno())

    return client_config_path, model_path, meta_data_path
