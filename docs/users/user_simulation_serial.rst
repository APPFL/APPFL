Simulating PPFL (Serial)
========================

In this section, we describe how to simulate PPFL on a single machine by running different FL clients serially.

.. note::

    Serial simulation can only be used for synchronous FL algorithms.

First, load the configuration files for the client and server agents and specify the number of clients to simulate.

.. code-block:: python

    from omegaconf import OmegaConf

    num_clients = 10
    server_agent_config = OmegaConf.load(
        '<your_example_path>/configs/mnist/server_fedavg.yaml'
    )
    client_agent_configs = [
        OmegaConf.load('<your_example_path>/configs/mnist/client_1.yaml')
        for _ in range(num_clients)
    ]

Then, update certain fields in the server agent configuration to make it compatible with ``num_clients``, and customize the client agent configuration for each client.

.. code-block:: python

    # Update server agent configuration
    server_agent_config.server_configs.num_clients = num_clients

    # Update client agent configurations
    for i in range(num_clients):
        client_agent_configs[i].client_id = f'Client{i+1}'
        client_agent_configs[i].data_configs.dataset_kwnum_clients = num_clients
        client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
        client_agent_configs[i].data_configs.dataset_kwargs.visualization = True if i == 0 else False

Next, create the server agent and client agents using the configurations.

.. code-block:: python

    from appfl.agent import ClientAgent, ServerAgent

    server_agent = ServerAgent(server_agent_config=server_agent_config)
    client_agents = [
        ClientAgent(client_agent_config=client_agent_configs[i])
        for i in range(num_clients)
    ]

After creating the agents, start the FL process by having the server provide the general client configurations to all clients using the ``server_agent.get_client_configs()`` function. The clients load the configurations using the ``client_agent.load_config()`` function.

Then, the clients get the initial global model from the server using the ``server_agent.get_parameters()`` function. Note that we set ``serial_run=True`` in ``get_parameters()`` to tell the server that the clients run serially, so it should not wait for all clients to call this function before sending the global model, which would block the FL process. The clients load the global model using the ``client_agent.load_parameters()`` function.

.. code-block:: python

    # Get additional client configurations from the server
    client_config_from_server = server_agent.get_client_configs()
    for client_agent in client_agents:
        client_agent.load_config(client_config_from_server)

    # Load initial global model from the server
    init_global_model = server_agent.get_parameters(serial_run=True)
    for client_agent in client_agents:
        client_agent.load_parameters(init_global_model)

.. note::

    Does it seem confusing that the server sends client configurations to all the clients? This is because, in FL, we usually want certain client-side configurations, such as the local trainer and the ML model architecture, to be the same for all clients. It is therefore more convenient to specify those configurations on the server side to ensure uniformity, and then send them to all clients at the beginning of the FL experiment.


Optionally, the clients can send the number of their local training samples to the server (which is useful if the server performs weighted averaging), and the server records the sample size of each client using the ``server_agent.set_sample_size()`` function.

.. code-block:: python

    # [Optional] Send the number of local data samples to the server
    for i in range(num_clients):
        sample_size = client_agents[i].get_sample_size()
        server_agent.set_sample_size(
            client_id=client_agents[i].get_id(),
            sample_size=sample_size
        )

After the above initialization, start the FL training loop:

- ``server_agent.training_finished()`` returns ``True`` if the training is finished, i.e., the stopping criteria are met.
- ``client_agent.train()`` performs local training on the client side.
- ``client_agent.get_parameters()`` gets the local model parameters from the client side, which can be a model state dictionary, model gradients, a compressed model, etc., depending on the training and compressor settings.
- ``server_agent.global_update()`` takes the local model from one client and returns the updated global model once it is ready. For synchronous FL, the server has to receive the local models from all clients before updating the global model. Therefore, to avoid blocking the FL process, the ``blocking`` argument is set to ``False``, and the function returns a ``Future`` object that will be resolved once the server has received the local models from all clients.
- After ``server_agent.global_update()`` has been called ``num_clients`` times, the global model is updated and all the ``Future`` objects are resolved. The clients can then load the new global model using the ``client_agent.load_parameters()`` function.

.. code-block:: python

    while not server_agent.training_finished():
        new_global_models = []
        for client_agent in client_agents:
            # Client local training
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            # "Send" local model to server and get a Future object for the new global model
            # The Future object will be resolved when the server receives local models from all clients
            new_global_model_future = server_agent.global_update(
                client_id=client_agent.get_id(),
                local_model=local_model,
                blocking=False,
                **metadata
            )
            new_global_models.append(new_global_model_future)
        # Load the new global model from the server
        for client_agent, new_global_model_future in zip(client_agents, new_global_models):
            client_agent.load_parameters(new_global_model_future.result())
