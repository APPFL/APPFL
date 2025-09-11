import os
import json
import torch
import shutil
from omegaconf import DictConfig
from peft import PeftModel, LoraConfig
from safetensors.torch import load_file
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from fed_sb.fed.fed_agg import (
    aggregate_models_fed_it,
    aggregate_models_fed_sb,
    aggregate_models_fedex,
    aggregate_models_ffa,
)
from fed_sb.utils.initialization_utils import find_and_initialize


class FedSBAggregator(BaseAggregator):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.aggregator_configs = aggregator_configs
        self.logger = logger

    def get_parameters(self, **kwargs):
        raise NotImplementedError("FedSB does not support get_parameters.")

    def aggregate(self, local_models: Dict[Union[str, int], str], **kwargs) -> Dict:
        client_ids = list(local_models.keys())
        aggregator_kwargs = {
            key: value[client_ids[0]] for key, value in kwargs.items()
        }  # assume all clients have the same kwargs
        agg_type = aggregator_kwargs["agg_type"]

        # Read through the local model adapters
        adapter_state_dicts = []
        for local_model_path in local_models.values():
            adapter_path = os.path.join(local_model_path, "adapter_model.safetensors")
            adapter = load_file(adapter_path, device=self.aggregator_configs.device)
            adapter_state_dicts.append(adapter)

        client_models = []

        for adapter_state_dict in adapter_state_dicts:
            new_weights = {}
            for key, value in adapter_state_dict.items():
                if agg_type == "fed-sb":
                    if "lora_A" in key:
                        new_key = key.replace("lora_A", "lora_A.default")
                    elif "lora_B" in key:
                        new_key = key.replace("lora_B", "lora_B.default")
                    elif "lora_latent" in key:
                        new_key = key.replace("_lora_latent", ".default_lora_latent")
                    else:
                        new_key = key
                else:
                    if "lora_A" in key:
                        new_key = key.replace("lora_A", "lora_A.default")
                    elif "lora_B" in key:
                        new_key = key.replace("lora_B", "lora_B.default")
                    else:
                        new_key = key

                new_weights[new_key] = value
            client_models.append(new_weights)

        if agg_type == "fed-sb":
            global_model, tokenizer = self.load_model_with_lora_sb(
                aggregator_kwargs["model_name"],
                new_weights,
                list(local_models.values())[0],
                aggregator_kwargs,
            )
        else:
            global_model, tokenizer = self.load_model_with_lora(
                aggregator_kwargs["model_name"],
                list(local_models.values())[0],
                aggregator_kwargs,
            )

        if agg_type == "fedex":
            global_model = aggregate_models_fedex(
                global_model, client_models, aggregator_kwargs
            )
        elif agg_type == "fed-it":
            global_model = aggregate_models_fed_it(global_model, client_models)
        elif agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)
        elif agg_type == "fed-sb":
            global_model = aggregate_models_fed_sb(global_model, client_models)

        if agg_type == "fed-sb":
            for param in global_model.parameters():
                param.data = param.data.contiguous()

        save_directory_final_model = os.path.join(
            self.aggregator_configs.global_model_dir, "final_model"
        )
        global_model.save_pretrained(save_directory_final_model)

        # Clean up final model directories used for aggregation
        for client_model in local_models.values():
            if os.path.exists(client_model):
                shutil.rmtree(client_model)
                self.logger.info(f"Deleted {client_model}")

        global_model = global_model.merge_and_unload()

        save_directory_merged_model = os.path.join(
            self.aggregator_configs.global_model_dir, "merged_model"
        )
        global_model.save_pretrained(save_directory_merged_model)

        save_directory_tokenizer = os.path.join(
            self.aggregator_configs.global_model_dir, "merged_model"
        )
        tokenizer.save_pretrained(save_directory_tokenizer)
        self.logger.info(f"Model saved to {save_directory_merged_model}")
        return {"merged_model_path": save_directory_merged_model}

    def load_model_with_lora_sb(
        self, base_model_name, lora_weights, lora_weights_path, args
    ):
        """
        Load a base model with LoRA weights

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            lora_weights_path (str): Path to saved LoRA weights directory

        Returns:
            model: Combined model with LoRA weights
            tokenizer: Associated tokenizer
        """
        # 1. Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map={"": "cuda"}, torch_dtype=torch.bfloat16
        )

        # 2. Load the tokenizer
        if "llama" in base_model_name:
            if "Llama-3" in base_model_name:
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": "cuda"},
                )
            else:
                tokenizer = LlamaTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": "cuda"},
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": "cuda"},
            )

        tokenizer.pad_token = tokenizer.eos_token

        # 3. Load and apply LoRA weights
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights_path,
            device_map={"": "cuda"},
        )

        reconstr_config = self.aggregator_configs.reconstruction_configs

        with open(os.path.join(lora_weights_path, "adapter_config.json")) as f:
            lora_config_dict = json.load(f)
            lora_config = LoraConfig(**lora_config_dict)

        adapter_name = "default"
        peft_config_dict = {adapter_name: lora_config}
        reconstr_config["svd"]["rank"] = args["lora_r"]

        find_and_initialize(
            model,
            peft_config_dict,
            adapter_name=adapter_name,
            reconstr_type="svd",
            writer=None,
            reconstruct_config=reconstr_config,
        )

        model_state_dict = model.state_dict()
        for key in model_state_dict.keys():
            if ("lora_A" in key) or ("lora_B" in key):
                model_state_dict[key] = lora_weights[key]

        model.load_state_dict(model_state_dict)

        return model, tokenizer

    def load_model_with_lora(self, base_model_name, lora_weights_path, args):
        """
        Load a base model with LoRA weights

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            lora_weights_path (str): Path to saved LoRA weights directory

        Returns:
            model: Combined model with LoRA weights
            tokenizer: Associated tokenizer
        """
        # 1. Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map={"": "cuda"}, torch_dtype=torch.bfloat16
        )

        if "llama" in base_model_name:
            if "Llama-3" in base_model_name:
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": "cuda"},
                )
            else:
                tokenizer = LlamaTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": "cuda"},
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": "cuda"},
            )

        tokenizer.pad_token = tokenizer.eos_token

        # 3. Load and apply LoRA weights
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights_path,
            device_map={"": "cuda"},
        )

        return model, tokenizer
