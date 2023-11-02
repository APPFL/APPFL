import copy
import json
import torch
from torch.utils import data

class Dataset(data.Dataset):
    """This class provides a simple way to define client dataset for supervised learning.
    This is derived from ``torch.utils.data.Dataset`` so that can be loaded to ``torch.utils.data.DataLoader``.
    Users may also create their own dataset class derived from this for more data processing steps.

    An empty ``Dataset`` class is created if no argument is given (i.e., ``Dataset()``).

    Args:
        data_input (torch.FloatTensor): optional data inputs
        data_label (torch.Tensor): optional data ouputs (or labels)
    """

    def __init__(
        self,
        data_input: torch.FloatTensor = torch.FloatTensor(),
        data_label: torch.Tensor = torch.Tensor(),
    ):
        self.data_input = data_input
        self.data_label = data_label

    def __len__(self):
        """This returns the sample size."""
        return len(self.data_label)

    def __getitem__(self, idx):
        """This returns a sample point for given ``idx``."""
        return self.data_input[idx], self.data_label[idx]

class AlpacaDataset(data.Dataset):
    """
    This class provides a class to load a json file into Alpaca-formated dataset.
    For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html
    """
    def __init__(self, data_path, tokenizer, partition="train", max_words=224):
        self.instructions = json.load(open(data_path))
        if partition == "train":
            self.instructions = self.instructions
        else:
            self.instructions = self.instructions[:200]

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.prompt_dict = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        instruct = self.instructions[index]
        if instruct.get("input", "") == "":
            prompt = self.prompt_dict["prompt_no_input"].format_map(instruct)
        else:
            prompt = self.prompt_dict["prompt_input"].format_map(instruct)
        labeled_prompt = prompt + instruct["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        labeled_prompt = self.tokenizer.encode(labeled_prompt)
        labeled_prompt.append(self.tokenizer.eos_token_id)
        labeled_prompt = torch.tensor(
            labeled_prompt, dtype=torch.int64
        )
        padding = self.max_words - labeled_prompt.shape[0]
        if padding > 0:
            labeled_prompt = torch.cat((labeled_prompt, torch.zeros(padding, dtype=torch.int64)-1)) # TODO: Check left or right padding
        elif padding < 0:
            labeled_prompt = labeled_prompt[:self.max_words]
        
        labels = copy.deepcopy(labeled_prompt)
        labels[:len(prompt)] = -1
        labeled_prompt_mask = labeled_prompt.ge(0)
        labels_mask = labels.ge(0)
        labeled_prompt[~labeled_prompt_mask] = 0
        labels[~labels_mask] = IGNORE_INDEX
        labeled_prompt_mask = labeled_prompt_mask.float()
        return {
            "input_ids": labeled_prompt,
            "labels": labels,
            "attention_mask": labeled_prompt_mask
        }

# TODO: This is very specific to certain data format.
def data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel):

    ## Check if "DataLoader" from PyTorch works.
    train_dataloader = data.DataLoader(train_datasets[0], batch_size=64, shuffle=False)

    for input, label in train_dataloader:

        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel

    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    for input, label in test_dataloader:

        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel
