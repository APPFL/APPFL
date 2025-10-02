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

        # Determine approach once with a flag
        is_file_based = isinstance(first_client_data, str)
        is_direct_content = isinstance(first_client_data, dict) and "adapter_weights" in first_client_data

        if not (is_file_based or is_direct_content):
            raise ValueError(f"Unexpected data format: {type(first_client_data)}")

        # Extract metadata from kwargs
        aggregator_kwargs = {
            key: value[client_ids[0]] for key, value in kwargs.items()
        }  # assume all clients have the same kwargs
        agg_type = aggregator_kwargs["agg_type"]

        # Save debug info for comparison
        approach_type = "file_based" if is_file_based else "direct_content"
        self._save_debug_keys_and_configs(local_models, aggregator_kwargs, approach_type)

        # Load adapter weights from both approaches and apply same transformation
        client_models = []
        for client_data in local_models.values():
            # Load adapter weights based on approach
            if is_file_based:
                # File-based: load from file path
                adapter_path = os.path.join(client_data, "adapter_model.safetensors")
                adapter_weights = load_file(adapter_path, device=self.aggregator_configs.device)
            else:  # is_direct_content
                # Direct content: get from client params and move to device
                adapter_weights = {
                    k: v.to(self.aggregator_configs.device)
                    for k, v in client_data["adapter_weights"].items()
                }

            # Apply same key transformation for both approaches
            new_weights = {}
            for key, value in adapter_weights.items():
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

        # Get adapter config (same logic for both approaches)
        if is_file_based:
            # From file
            first_model_path = list(local_models.values())[0]
            config_path = os.path.join(first_model_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    adapter_config = json.load(f)
            else:
                adapter_config = {}
        else:  # is_direct_content
            # From direct content
            adapter_config = list(local_models.values())[0].get("adapter_config", {})

        # Load model and run aggregation (same for both approaches)
        if agg_type == "fed-sb":
            # Pass lora_weights_path and transformed weights for file-based approach
            if is_file_based:
                lora_weights_path = list(local_models.values())[0]
                transformed_weights = client_models[0]  # Use transformed weights for file-based
            else:
                lora_weights_path = None
                transformed_weights = None

            global_model, tokenizer = self.load_model_with_lora_sb(
                aggregator_kwargs["model_name"],
                transformed_weights,
                adapter_config,
                aggregator_kwargs,
                lora_weights_path
            )
        else:
            # Pass lora_weights_path only for file-based approach
            lora_weights_path = list(local_models.values())[0] if is_file_based else None
            global_model, tokenizer = self.load_model_with_lora(
                aggregator_kwargs["model_name"],
                adapter_config,
                aggregator_kwargs,
                lora_weights_path
            )

        # Run aggregation algorithm
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

        # Save final model
        save_directory_final_model = os.path.join(
            self.aggregator_configs.global_model_dir, "final_model"
        )
        global_model.save_pretrained(save_directory_final_model)

        # Clean up files only if using file-based approach
        if is_file_based:
            for client_model in local_models.values():
                if os.path.exists(client_model):
                    shutil.rmtree(client_model)
                    if self.logger:
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
        if self.logger:
            self.logger.info(f"Model saved to {save_directory_merged_model}")
        return {"merged_model_path": save_directory_merged_model}

    def load_model_with_lora_sb(
        self, base_model_name, lora_weights, adapter_config, args, lora_weights_path=None
    ):
        """
        Load a base model with LoRA weights for Fed-SB (unified function)

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            lora_weights (dict): LoRA weights dictionary (for direct approach)
            adapter_config (dict): Adapter configuration dictionary
            args: Arguments containing LoRA parameters
            lora_weights_path (str, optional): Path to LoRA weights for file-based approach

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

        if lora_weights_path:
            # File-based approach: load from path
            model = PeftModel.from_pretrained(
                base_model,
                lora_weights_path,
                device_map={"": "cuda"},
            )

            # Load config from file
            with open(os.path.join(lora_weights_path, "adapter_config.json")) as f:
                lora_config_dict = json.load(f)
                lora_config = LoraConfig(**lora_config_dict)
        else:
            # Direct approach: create PEFT model with config
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

        # Fed-SB specific initialization
        reconstr_config = self.aggregator_configs.reconstruction_configs
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

        # For file-based approach, apply the transformed weights
        if lora_weights_path and lora_weights:
            model_state_dict = model.state_dict()
            for key in model_state_dict.keys():
                if ("lora_A" in key) or ("lora_B" in key):
                    model_state_dict[key] = lora_weights[key]
            model.load_state_dict(model_state_dict)

        return model, tokenizer

    def load_model_with_lora(self, base_model_name, adapter_config, args, lora_weights_path=None):
        """
        Load a base model with LoRA weights (unified function for both direct config and file path)

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            adapter_config (dict): Adapter configuration dictionary
            args: Arguments containing LoRA parameters
            lora_weights_path (str, optional): Path to LoRA weights for file-based approach

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

        # 3. Create LoRA model
        if lora_weights_path:
            # File-based approach: load from path
            model = PeftModel.from_pretrained(
                base_model,
                lora_weights_path,
                device_map={"": "cuda"},
            )
        else:
            # Direct approach: create PEFT model with config
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

    def _save_debug_keys_and_configs(self, local_models, aggregator_kwargs, approach_type):
        """
        Save keys and configs for debugging/comparison purposes.
        """
        try:
            import os
            import json
            from datetime import datetime

            # Create debug directory
            debug_dir = os.path.join(os.getcwd(), "fedsb_debug_comparison")
            os.makedirs(debug_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = os.path.join(debug_dir, f"{approach_type}_keys_configs_{timestamp}.json")

            debug_info = {
                "approach": approach_type,
                "timestamp": timestamp,
                "aggregator_kwargs": aggregator_kwargs,
                "clients": {}
            }

            print(f"[SERVER DEBUG] Saving {approach_type} approach debug info...")

            for client_id, client_data in local_models.items():
                client_info = {
                    "client_id": client_id,
                    "data_type": type(client_data).__name__
                }

                if approach_type == "file_based":
                    # Extract info from file path
                    model_path = client_data
                    client_info["model_path"] = model_path

                    # Load safetensors to get keys
                    adapter_path = os.path.join(model_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_path):
                        adapter_weights = load_file(adapter_path, device="cpu")
                        client_info["adapter_keys"] = list(adapter_weights.keys())
                        client_info["adapter_shapes"] = {k: list(v.shape) for k, v in adapter_weights.items()}
                        client_info["total_params"] = sum(v.numel() for v in adapter_weights.values())
                        print(f"  Client {client_id}: {len(adapter_weights)} keys, {client_info['total_params']:,} params")
                    else:
                        client_info["adapter_keys"] = []
                        client_info["error"] = f"adapter_model.safetensors not found in {model_path}"

                    # Load config
                    config_path = os.path.join(model_path, "adapter_config.json")
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        client_info["adapter_config_keys"] = list(config.keys())
                        client_info["adapter_config"] = config
                        print(f"  Client {client_id}: {len(config)} config keys")
                    else:
                        client_info["adapter_config_keys"] = []
                        client_info["error"] = f"adapter_config.json not found in {model_path}"

                elif approach_type == "direct_content":
                    # Extract info from direct content
                    adapter_weights = client_data.get("adapter_weights", {})
                    adapter_config = client_data.get("adapter_config", {})

                    client_info["adapter_keys"] = list(adapter_weights.keys())
                    client_info["adapter_shapes"] = {k: list(v.shape) for k, v in adapter_weights.items()}
                    client_info["total_params"] = sum(v.numel() for v in adapter_weights.values())
                    client_info["adapter_config_keys"] = list(adapter_config.keys())
                    client_info["adapter_config"] = adapter_config

                    print(f"  Client {client_id}: {len(adapter_weights)} keys, {client_info['total_params']:,} params")
                    print(f"  Client {client_id}: {len(adapter_config)} config keys")

                debug_info["clients"][client_id] = client_info

            # Save to file
            with open(debug_file, "w") as f:
                json.dump(debug_info, f, indent=2)

            print(f"[SERVER DEBUG] Saved debug info to: {debug_file}")

        except Exception as e:
            print(f"[SERVER DEBUG ERROR] Failed to save debug info: {e}")
