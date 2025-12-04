import time
import os
import deepspeed
import torch
from appfl.agent import ClientAgent
from appfl.comm.utils.client_utils import (
    load_global_model,
    send_local_model,
)
import torch.distributed as dist


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
    if (
        hasattr(client_agent_config, "data_readiness_configs")
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        )
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "remedy_action",
        )
        and client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
    ):
        client_agent.adapt_data(client_config=client_agent_config)
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
    if (
        hasattr(client_agent_config, "data_readiness_configs")
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        )
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "remedy_action",
        )
        and client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
    ):
        client_agent.adapt_data(client_config=client_agent_config)
    return None, {
        "data_readiness": client_agent.generate_readiness_report(client_agent_config)
    }


def train_executor(
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    task_sent_time = float(client_agent_config.start_time)
    total_task_sent_time = time.time() - task_sent_time
    print(f"serve to client time: {total_task_sent_time}")

    task_start_time = time.time()
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    if (
        hasattr(client_agent_config, "data_readiness_configs")
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        )
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "remedy_action",
        )
        and client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
    ):
        client_agent.adapt_data(client_config=client_agent_config)

    donwload_model_start_time = time.time()
    if model is not None:
        model = load_global_model(client_agent.client_agent_config, model)
        client_agent.load_parameters(model)
    total_model_download_time = time.time() - donwload_model_start_time
    print(f"total download model time: {total_model_download_time}")

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
    total_task_execution_time = time.time() - task_start_time

    meta_data_local["end_time"] = time.time()
    meta_data_local["server_to_client_time"] = total_task_sent_time
    meta_data_local["total_model_download_time"] = total_model_download_time
    meta_data_local["total_execution_time"] = total_task_execution_time
    meta_data_local["total_pre_val_time"] = client_agent.trainer.total_pre_val_time
    meta_data_local["total_forward_time"] = client_agent.trainer.total_forward_time
    meta_data_local["total_backward_time"] = client_agent.trainer.total_backward_time
    meta_data_local["total_val_time"] = client_agent.trainer.total_val_time

    return local_model, meta_data_local


def get_sample_size_executor_ds(
    client_agent_config=None,
    **kwargs,
):
    # local_rank = int(os.environ['MPI_LOCALRANKID'])
    local_rank = int(
        os.environ.get("PMIX_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    )
    # print(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

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

        # torch.distributed.barreir()

        return None, meta_data

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def get_ds_config(client_agent_config, world_size):
    accumulation_steps = client_agent_config.train_configs.gradient_accumulation_steps
    batch_size_per_gpu = client_agent_config.train_configs.train_batch_size
    learning_rate = client_agent_config.train_configs.learning_rate

    ds_config = {
        "gradient_accumulation_steps": accumulation_steps,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "train_batch_size": accumulation_steps * batch_size_per_gpu * world_size,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "weight_decay": 0.01,
                "torch_adam": True,
                "adam_w_mode": True,
            },
        },
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "wall_clock_breakdown": True,
        "memory_breakdown": True,
        "dump_state": True,
        "flops_profiler": {
            "enabled": True,
            "profile_step": 5,  # Profile first step
            "module_depth": -1,  # Full depth
            "top_modules": 3,
            "detailed": True,
            "output_file": None,  # Print to stdout
        },
        # Optional: Control output frequency
        "steps_per_print": 1,  # Print monitoring info every X steps
    }

    if (
        client_agent_config.train_configs.clip_grad
        and client_agent_config.train_configs.clip_norm == 2.0
    ):
        ds_config["gradient_clipping"] = client_agent_config.train_configs.clip_value

    ds_stage = client_agent_config.ds_configs.zero_stages
    if ds_stage == 1:
        ds_config["zero_optimization"] = {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e7,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "contiguous_gradients": True,
        }
    elif ds_stage == 2:
        ds_config["zero_optimization"] = {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        }
    elif ds_stage == 3:
        ds_config["zero_optimization"] = {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 3e9,
            "stage3_max_reuse_distance": 3e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        }

    if client_agent_config.ds_configs.activation_checkpointing:
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 2,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        }

    return ds_config


def train_executor_ds(
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    local_rank_vars = [
        "MPI_LOCALRANKID",
        "PALS_LOCAL_RANKID",
        "PMIX_LOCAL_RANK",
        "SLURM_LOCALID",
    ]
    for var in local_rank_vars:
        if var in os.environ:
            local_rank = int(os.environ[var])
            os.environ["LOCAL_RANK"] = str(local_rank)
            break

    # set cuda or xpu according to the device type
    if client_agent_config.train_configs.device == "cuda":
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    elif client_agent_config.train_configs.device == "xpu":
        import intel_extension_for_pytorch as ipex # noqa F401
        if torch.xpu.is_available():
            torch.xpu.set_device(local_rank)

    deepspeed.init_distributed()
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()

    try:
        start_time = time.time()
        task_sent_time = float(client_agent_config.start_time)
        total_task_sent_time = time.time() - task_sent_time
        print(f"serve to client time: {total_task_sent_time}")

        download_model_start_time = time.time()

        if model is not None:
            if global_rank == 0:
                print(f"Rank 0/{world_size}: Starting model download via extract...")
                model = load_global_model(client_agent_config, model)
                total_model_download_time = time.time() - download_model_start_time
                print(
                    f"Rank 0: Model download completed in {total_model_download_time:.2f}s"
                )

        dist.barrier()

        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_agent.global_rank = global_rank
        client_agent.trainer.global_rank = global_rank
        client_agent.world_size = world_size
        client_agent.trainer.world_size = world_size

        if model is not None:
            if global_rank == 0:
                # Rank 0: Load the extracted model's state_dict
                client_agent.load_parameters(model)
                print(
                    f"Rank 0: Parameters loaded, broadcasting to {world_size - 1} other ranks..."
                )
                broadcast_start = time.time()

            if client_agent_config.train_configs.device == "cuda":
                device = torch.device(f"cuda:{local_rank}")
                client_agent.model = client_agent.model.to(device)
            elif client_agent_config.train_configs.device == "xpu":
                device = torch.device(f"xpu:{local_rank}")
                client_agent.model = client_agent.model.to(device)

            # Broadcast model weights from rank 0 to all others
            for param in client_agent.model.parameters():
                dist.broadcast(param.data, src=0)

            for buffer in client_agent.model.buffers():
                dist.broadcast(buffer, src=0)

            if global_rank == 0:
                broadcast_time = time.time() - broadcast_start
                total_time = time.time() - download_model_start_time
                print(f"Rank 0: Broadcast completed in {broadcast_time:.2f}s")
                print(f"Rank 0: Total model loading time: {total_time:.2f}s")

        dist.barrier()

        ds_config = get_ds_config(client_agent_config, world_size)

        client_agent.trainer.model_engine, client_agent.trainer.optimizer, _, _ = (
            deepspeed.initialize(
                model=client_agent.model,
                model_parameters=client_agent.model.parameters(),
                config=ds_config,
            )
        )

        client_agent.train(**meta_data)

        local_model = None
        meta_data_local = None

        if client_agent_config.ds_configs.zero_stages == 3:
            with deepspeed.zero.GatheredParameters(
                client_agent.trainer.model_engine.module.parameters(), modifier_rank=0
            ):
                if global_rank == 0:
                    local_model = client_agent.trainer.model_engine.module.state_dict()
        else:
            if global_rank == 0:
                local_model = client_agent.trainer.model_engine.module.state_dict()

        if global_rank == 0:
            model_save_path = os.path.join(
                client_agent_config.save_dir, f"round{meta_data['round']}.pth"
            )
            torch.save(local_model, model_save_path)
            if isinstance(local_model, tuple):
                local_model, meta_data_local = local_model
            else:
                meta_data_local = {}
            for k in local_model:
                local_model[k] = local_model[k].cpu()
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
            end_time = time.time()

            meta_data_local["end_time"] = end_time
            meta_data_local["server_to_client_time"] = total_task_sent_time
            meta_data_local["total_model_download_time"] = total_model_download_time
            meta_data_local["total_execution_time"] = end_time - start_time
            meta_data_local["total_pre_val_time"] = (
                client_agent.trainer.total_pre_val_time
            )
            meta_data_local["total_forward_time"] = (
                client_agent.trainer.total_forward_time
            )
            meta_data_local["total_backward_time"] = (
                client_agent.trainer.total_backward_time
            )
            meta_data_local["total_val_time"] = client_agent.trainer.total_val_time
            if client_agent_config.train_configs.mode == "epoch":
                meta_data_local["avg_epoch_loss"] = (
                    client_agent.trainer.avg_epoch_loss_dict
                )
            elif client_agent_config.train_configs.mode == "step":
                meta_data_local["avg_step_loss"] = client_agent.trainer.avg_step_loss

        return local_model, meta_data_local

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
