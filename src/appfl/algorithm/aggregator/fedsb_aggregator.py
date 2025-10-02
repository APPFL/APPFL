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

    def aggregate(self, local_models: Dict[Union[str, int], Union[str, Dict]], **kwargs) -> Dict:
        client_ids = list(local_models.keys())
        first_client_data = list(local_models.values())[0]

        # Debug information
        self.logger.info(f"Aggregator received local_models type: {type(first_client_data)}")
        self.logger.info(f"Aggregator received kwargs keys: {list(kwargs.keys())}")
        if isinstance(first_client_data, dict):
            self.logger.info(f"First client data keys: {list(first_client_data.keys())}")

        # Handle both old format (string paths) and new format (parameter dicts)
        if isinstance(first_client_data, str):
            # Old format - file path (backward compatibility)
            self.logger.info("Using file-based aggregation (backward compatibility)")
            return self._aggregate_from_files(local_models, **kwargs)
        elif isinstance(first_client_data, dict):
            # New format - direct parameter transfer
            # Check if it's the new format with adapter_weights
            if "adapter_weights" in first_client_data:
                self.logger.info("Using parameter-based aggregation (new format)")
                return self._aggregate_from_params(local_models, **kwargs)
            else:
                # This might be some other dict format, fall back to file-based
                self.logger.info("Unknown dict format, falling back to file-based aggregation")
                return self._aggregate_from_files(local_models, **kwargs)
        else:
            raise ValueError(f"Unexpected data format: {type(first_client_data)}")

    def _aggregate_from_params(self, local_models: Dict[Union[str, int], Dict], **kwargs) -> Dict:
        client_ids = list(local_models.keys())
        # Extract metadata from kwargs (passed from server)
        aggregator_kwargs = {
            key: value[client_ids[0]] for key, value in kwargs.items()
        }  # assume all clients have the same kwargs
        agg_type = aggregator_kwargs["agg_type"]

        # Get first client's parameters for adapter config
        first_client_params = list(local_models.values())[0]

        # Extract adapter weights directly from client data
        adapter_state_dicts = []
        for client_params in local_models.values():
            adapter_weights = client_params["adapter_weights"]
            # Debug: Log what keys we're getting from the trainer
            self.logger.info(f"Original adapter keys from trainer (first 3): {list(adapter_weights.keys())[:3]}")
            # Move weights to the target device
            adapter_weights_on_device = {
                k: v.to(self.aggregator_configs.device) for k, v in adapter_weights.items()
            }
            adapter_state_dicts.append(adapter_weights_on_device)

        client_models = []

        for adapter_state_dict in adapter_state_dicts:
            # For direct parameter transfer, we might need to use the weights as-is
            # Let's try without transformation first to see what the aggregation function expects
            new_weights = adapter_state_dict
            # Debug: Log keys before any transformation
            self.logger.info(f"Using adapter keys as-is (first 3): {list(new_weights.keys())[:3]}")
            client_models.append(new_weights)

        if agg_type == "fed-sb":
            global_model, tokenizer = self.load_model_with_lora_sb(
                aggregator_kwargs["model_name"],
                new_weights,
                first_client_params["adapter_config"],
                aggregator_kwargs,
            )
        else:
            global_model, tokenizer = self.load_model_with_lora(
                aggregator_kwargs["model_name"],
                first_client_params["adapter_config"],
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

    def _aggregate_from_files(self, local_models: Dict[Union[str, int], str], **kwargs) -> Dict:
        """
        Original aggregation method that works with file paths (backward compatibility).
        """
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
                        new_key = key.replace("_lora_latent", ".default_lora_latent_mapping")
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
            global_model, tokenizer = self.load_model_with_lora_sb_from_path(
                aggregator_kwargs["model_name"],
                new_weights,
                list(local_models.values())[0],
                aggregator_kwargs,
            )
        else:
            global_model, tokenizer = self.load_model_with_lora_from_path(
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
        self, base_model_name, lora_weights, adapter_config, args
    ):
        """
        Load a base model with LoRA weights

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            lora_weights (dict): LoRA weights dictionary
            adapter_config (dict): Adapter configuration dictionary

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

        # 3. Create LoRA config from adapter_config and create PEFT model
        reconstr_config = self.aggregator_configs.reconstruction_configs

        if adapter_config:
            lora_config = LoraConfig(**adapter_config)
        else:
            # Fallback if no config provided
            lora_config = LoraConfig(
                r=args["lora_r"],
                lora_alpha=args["lora_alpha"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
            )

        # Create PEFT model with temporary adapter
        from peft import get_peft_model
        model = get_peft_model(base_model, lora_config)

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

        # Note: We don't load the lora_weights here since the aggregation algorithm
        # will handle setting the proper aggregated weights

        return model, tokenizer

    def load_model_with_lora(self, base_model_name, adapter_config, args):
        """
        Load a base model with LoRA weights

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            adapter_config (dict): Adapter configuration dictionary

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

        # 3. Create LoRA config and PEFT model
        if adapter_config:
            lora_config = LoraConfig(**adapter_config)
        else:
            # Fallback if no config provided
            lora_config = LoraConfig(
                r=args["lora_r"],
                lora_alpha=args["lora_alpha"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
            )

        # Create PEFT model
        from peft import get_peft_model
        model = get_peft_model(base_model, lora_config)

        return model, tokenizer

    def load_model_with_lora_sb_from_path(
        self, base_model_name, lora_weights, lora_weights_path, args
    ):
        """
        Load a base model with LoRA weights from file path (backward compatibility).
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

    def load_model_with_lora_from_path(self, base_model_name, lora_weights_path, args):
        """
        Load a base model with LoRA weights from file path (backward compatibility).
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
