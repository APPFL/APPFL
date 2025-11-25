"""
Loads a Hugging Face large language model (LLM) using the transformers library based on the specified model name.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_hf_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model