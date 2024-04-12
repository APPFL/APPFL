def globus_compute_client_entry_point(
    task_name="N/A",
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    from appfl.agent import APPFLClientAgent
    from appfl.comm.globus_compute.utils.client_utils import load_global_state
    
    client_agent = APPFLClientAgent(client_agent_config=client_agent_config)
    if model is not None:
        model = load_global_state(model)
        client_agent.load_parameters(model)

    if task_name == "get_sample_size":
        return {
            "sample_size": client_agent.get_sample_size()
        }
    
    elif task_name == "train":
        client_agent.train()
        local_model = client_agent.get_parameters()

    
    