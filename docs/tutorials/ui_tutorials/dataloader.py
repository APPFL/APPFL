def get_data(**kwargs):
    # Import necessary libraries
    import torch
    from torch.utils.data import Dataset
    ......

    # Write code to load your local data
    ......

    # Return PyTorch Dataset
    return Dataset(local_data, local_labels)
