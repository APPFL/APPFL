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
    from appfl.agent import ClientAgent
    from appfl.comm.globus_compute.utils.client_utils import load_global_model, send_local_model
    
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    if model is not None:
        model = load_global_model(client_agent.client_agent_config, model)
        client_agent.load_parameters(model)

    if task_name == "get_sample_size":
        return None, {
            "sample_size": client_agent.get_sample_size()
        }
    
    elif task_name == "data_readiness_report":
        return None,{
            "data_readiness": client_agent.generate_readiness_report(client_agent_config)
        }
    
    elif task_name == "train":
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, meta_data_local = local_model
        else:
            meta_data_local = {}
        local_model = send_local_model(
            client_agent.client_agent_config, 
            local_model,
            meta_data["local_model_key"] if "local_model_key" in meta_data else None,
            meta_data["local_model_url"] if "local_model_url" in meta_data else None,
        )
        return local_model, meta_data_local
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented in the client endpoint.")