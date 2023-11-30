def get_data(cfg, client_idx: int, mode="train"):
    """
    Note
    ----
    The dataloader assumes you have originally partitioned the data and stored in the the desired position.
    The `superglue_partition.py` script could help to partition the dataset and store the client splits in a user-specified data root path with desired format.
    Then replace the `data_root_path` variable with your (relative/absolute) data root path.
    """
    from transformers import AutoTokenizer
    from appfl.misc.data import AlpacaDataset
    data_root_path = "/projects/bbvf/zl52/globus-compute-endpoint/superglue_partitioned_data"
    data_path = f"{data_root_path}/{cfg.custom_configs.dataset}/{client_idx}/{mode}.json"
    tokenizer = AutoTokenizer.from_pretrained("/projects/bbke/zl52/llama2/llama/models_hf/7B_chat")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return AlpacaDataset(data_path, tokenizer, cfg.custom_configs.training_configs.max_words)
