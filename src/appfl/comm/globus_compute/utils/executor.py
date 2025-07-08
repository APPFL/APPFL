import time
import os
import importlib
import deepspeed
import torch
from appfl.agent import ClientAgent
from appfl.comm.globus_compute.utils.client_utils import (
    load_global_model,
    send_local_model,
)


def get_sample_size_executor(
    client_agent_config=None,
    **kwargs,
):  
    task_sent_time = float(client_agent_config.start_time)
    total_task_sent_time = time.time() - task_sent_time
    print(f"serve to client time: {total_task_sent_time}")
    task_execution_start_time = time.time()
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    sample_size = client_agent.get_sample_size()
    # return None, {"sample_size": client_agent.get_sample_size()}
    total_task_execution_time = time.time() - task_execution_start_time
    print(f"total get sample time: {total_task_execution_time}")
    return None, {
        "sample_size": sample_size, 
        "end_time": time.time(), 
        "server_to_client_time": total_task_sent_time,
        "total_model_download_time": "n/a",
        "total_execution_time": total_task_execution_time,
        "total_pre_val_time": "n/a",
        "total_forward_time": "n/a",
        "total_backward_time": "n/a",
        "total_val_time": "n/a",
    }


def data_readiness_report_executor(
    client_agent_config=None,
    **kwargs,
):
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    return None, {
        "data_readiness": client_agent.generate_readiness_report(client_agent_config)
    }


def train_executor(
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    # client_agent = ClientAgent(client_agent_config=client_agent_config)
    task_sent_time = float(client_agent_config.start_time)
    total_task_sent_time = time.time() - task_sent_time
    print(f"serve to client time: {total_task_sent_time}")
    
    donwload_model_start_time = time.time()
    if model is not None:
        # model = load_global_model(client_agent.client_agent_config, model)
        # client_agent.load_parameters(model)
        model = load_global_model(client_agent_config, model)
    total_model_download_time = time.time() - donwload_model_start_time
    print(f"total donwload model time: {total_model_download_time}")
    
    training_start_time = time.time()  
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_agent.load_parameters(model)

    client_agent.train(**meta_data)
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
    total_task_execution_time = time.time() - training_start_time

    meta_data_local['end_time'] = time.time()
    meta_data_local['server_to_client_time'] = total_task_sent_time
    meta_data_local['total_model_download_time'] = total_model_download_time
    meta_data_local['total_execution_time'] = total_task_execution_time
    meta_data_local['total_pre_val_time'] = client_agent.trainer.total_pre_val_time
    meta_data_local['total_forward_time'] = client_agent.trainer.total_forward_time
    meta_data_local['total_backward_time'] = client_agent.trainer.total_backward_time
    meta_data_local['total_val_time'] = client_agent.trainer.total_val_time
    return local_model, meta_data_local

def get_sample_size_executor_ds(
    client_agent_config=None,
    **kwargs,
):  
    # local_rank = int(os.environ['MPI_LOCALRANKID'])
    local_rank = int(os.environ.get('PMIX_LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    # print(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    deepspeed.init_distributed()
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()

    try:
        task_sent_time = float(client_agent_config.start_time)
        total_task_sent_time = time.time() - task_sent_time
        print(f"serve to client time: {total_task_sent_time}")
        task_execution_start_time = time.time()
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        sample_size = client_agent.get_sample_size()
        total_task_execution_time = time.time() - task_execution_start_time
        print(f"total get sample time: {total_task_execution_time}")

        meta_data = None
        if global_rank == 0:
            meta_data = {
                "sample_size": sample_size * world_size, 
                "end_time": time.time(), 
                "server_to_client_time": total_task_sent_time,
                "total_model_download_time": "n/a",
                "total_execution_time": total_task_execution_time,
                "total_pre_val_time": "n/a",
                "total_forward_time": "n/a",
                "total_backward_time": "n/a",
                "total_val_time": "n/a",
            }
        
        #torch.distributed.barreir()
        
        return None, meta_data
        
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

def get_ds_config(client_agent_config, world_size):
    accumulation_steps = client_agent_config.train_configs.gradient_accumulation_steps
    batch_size_per_gpu = client_agent_config.train_configs.train_batch_size

    ds_config = {
        "gradient_accumulation_steps": accumulation_steps,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "train_batch_size": accumulation_steps * batch_size_per_gpu * world_size,
        "steps_per_print": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "torch_adam": True,
                "adam_w_mode": True
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": client_agent_config.ds_configs.zero_stages,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 2,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        "wall_clock_breakdown": False
    }

    if client_agent_config.train_configs.clip_grad == True and client_agent_config.train_configs.clip_norm == 2.0:
        ds_config["gradient_clipping"] = client_agent_config.train_configs.clip_value

    return ds_config

def train_executor_ds(
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    # local_rank = int(os.environ['MPI_LOCALRANKID'])
    local_rank = int(os.environ.get('PMIX_LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    os.environ['LOCAL_RANK'] = str(local_rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    deepspeed.init_distributed() #TODO backend specification: nccl and rccl and 1ccl?
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
    
    try:
        start_time = time.time()
        task_sent_time = float(client_agent_config.start_time)
        total_task_sent_time = time.time() - task_sent_time
        print(f"serve to client time: {total_task_sent_time}")

        donwload_model_start_time = time.time()
        if model is not None:
            model = load_global_model(client_agent_config, model)
        total_model_download_time = time.time() - donwload_model_start_time
        print(f"total donwload model time: {total_model_download_time}")
            
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_agent.global_rank = global_rank
        client_agent.world_size = world_size
        client_agent.load_parameters(model)

        ds_config = get_ds_config(client_agent_config, world_size)
        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, client_agent_config.train_configs.optim), (
            f"Optimizer {client_agent_config.train_configs.optim} not found in torch.optim"
        )
        optimizer = getattr(optim_module, client_agent_config.train_configs.optim)(
            client_agent.model.parameters(), **client_agent_config.train_configs.optim_args
        )

        client_agent.trainer.model_engine, client_agent.trainer.optimizer, _, _ = deepspeed.initialize(
            model=client_agent.model,
            optimizer=optimizer,
            model_parameters=client_agent.model.parameters(),
            config=ds_config,
        )

        client_agent.trainer.global_rank = global_rank

        client_agent.train(**meta_data)

        #Check everyone is done?

        local_model = None
        meta_data_local = None
        if global_rank == 0:
            client_agent.trainer.model = client_agent.trainer.model_engine.module
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, meta_data_local = local_model
            else:
                meta_data_local = {}
            for k in local_model:
                local_model[k] = local_model[k].cpu()
            local_model = send_local_model(
                client_agent.client_agent_config,
                local_model,
                meta_data["local_model_key"] if "local_model_key" in meta_data else None,
                meta_data["local_model_url"] if "local_model_url" in meta_data else None,
            )
            end_time = time.time()

            meta_data_local['end_time'] = end_time
            meta_data_local['server_to_client_time'] = total_task_sent_time
            meta_data_local['total_model_download_time'] = total_model_download_time
            meta_data_local['total_execution_time'] = end_time - start_time
            meta_data_local['total_pre_val_time'] = client_agent.trainer.total_pre_val_time
            meta_data_local['total_forward_time'] = client_agent.trainer.total_forward_time
            meta_data_local['total_backward_time'] = client_agent.trainer.total_backward_time
            meta_data_local['total_val_time'] = client_agent.trainer.total_val_time
        
        #torch.distributed.barreir()
        return local_model, meta_data_local
    
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()