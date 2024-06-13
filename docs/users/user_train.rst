Training PPFL
=============

To run PPFL with decentralized data on multiple machines, we use `gRPC <https://grpc.io/docs/>`_ that  allows clients from different platforms to seamlessly connect to the server for federated learning. This contrasts with MPI where all clients and servers should reside in the same cluster.

Launch a gRPC server
--------------------

The server needs to load configuration from a configuration file, then we can create a gRPC FL serverto listen to request from clients by calling the ``serve()`` function from ``appfl.communicator.grpc`` module.

.. code-block:: python

    from omegaconf import OmegaConf
    from appfl.agent import APPFLServerAgent
    from appfl.communicator.grpc import GRPCServerCommunicator, serve

    server_agent_config = OmegaConf.load('<path_to_config_file>.yaml')
    server_agent = APPFLServerAgent(server_agent_config=server_agent_config)

    communicator = GRPCServerCommunicator(
        server_agent,
        max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
        logger=server_agent.logger,
    )

    serve(
        communicator,
        **server_agent_config.server_configs.comm_configs.grpc_configs,
    )

Launch a gRPC client
--------------------

The client also loads configuration from a configuration file, and then it starts the FL process by sending different requests to the server.

.. code-block:: python

    from omegaconf import OmegaConf
    from appfl.agent import APPFLClientAgent
    from appfl.communicator.grpc import GRPCClientCommunicator

    client_agent_config = OmegaConf.load('<path_to_config_file>.yaml')

    client_agent = APPFLClientAgent(client_agent_config=client_agent_config)
    client_communicator = GRPCClientCommunicator(
        client_id = client_agent.get_id(),
        **client_agent_config.comm_configs.grpc_configs,
    )

    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    # Send the number of local data to the server
    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)

    while True:
        client_agent.train()
        local_model = client_agent.get_parameters()
        new_global_model, metadata = client_communicator.update_global_model(local_model)
        if metadata['status'] == 'DONE':
            break
        if 'local_steps' in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
        client_agent.load_parameters(new_global_model)