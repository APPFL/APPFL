import torch
import torch.nn as nn
from transformers import OPTForCausalLM

def get_opt(model_name):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.seqlen = model.config.max_position_embeddings
    return model