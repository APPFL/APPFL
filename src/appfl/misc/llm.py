import os
import torch
from collections import OrderedDict
from transformers import LlamaForCausalLM

def load_model(model_name, quantization):
    """Function to load the model for causal text generation"""
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

def load_peft_model(model, peft_model_path):
    """Load the saved peft model into the based model."""
    peft_state_dict = torch.load(peft_model_path, map_location="cuda")
    model.load_state_dict(peft_state_dict, strict=False)
    return model

def get_peft_state_dict(model):
    grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_params.append(name)
    model_state_dict = model.state_dict()
    trainable_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k in grad_params:
            trainable_state_dict[k] = v
    return trainable_state_dict

def save_peft_model(model, save_name):
    grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_params.append(name)
    model_state_dict = model.state_dict()
    trainable_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k in grad_params:
            trainable_state_dict[k] = v
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    torch.save(trainable_state_dict, save_name)