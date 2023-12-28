def get_data(cfg, client_idx: int, mode='train'):
    import torch
    import numpy as np
    import torchvision
    import os.path as osp
    from appfl.misc.data import Dataset
    from torchvision.transforms import ToTensor

    # Prepare local dataset directory
    data_dir = cfg.clients[client_idx].data_dir
    local_dir = osp.join(data_dir, "RawData")
    data_raw = torchvision.datasets.MNIST(
        local_dir, download = True, 
        train = True if mode == 'train' else False, 
        transform= ToTensor()
    )
    
    if mode != 'train':
        split_data_raw = np.array_split(range(len(data_raw)), cfg.num_clients)
        data_input = []
        data_label = []
        
        for idx in split_data_raw[client_idx]:
            data_input.append(data_raw[idx][0].tolist())
            data_label.append(data_raw[idx][1])
        
        return Dataset(
                torch.FloatTensor(data_input),
                torch.tensor(data_label),
            )
    else:
        # Partition the dataset using Class Non-IID
        np.random.seed(42)
        # The following code partition the dataset by label so that each client only has a subset of the labels
        # The code is just a reform of the original code in examples/dataloader/utils/partition.py
        # training data for multiple clients
        Cmin = {1: 10, 2: 7, 3: 6, 4: 5, 5: 5, 6: 4, 7: 4, 'none': 3}       # minimum sample classes for each client
        Cmax = {1: 10, 2: 8, 3: 8, 4: 7, 5: 6, 6: 6, 7: 5, 'none': 5}       # maximum sample classes for each client

        # Split the dataset by label
        labels = []
        label_indices = {}
        for idx, (_, label) in enumerate(data_raw):
            if label not in label_indices:
                label_indices[label] = []
                labels.append(label)
            label_indices[label].append(idx)
        labels.sort()

        # Obtain the way to partition the dataset
        num_clients = cfg.num_clients
        while True:
            class_partition = {}    # number of partitions for each class of MNIST
            client_classes  = {}    # sample classes for different clients
            for i in range(num_clients):
                cmin = Cmin[num_clients] if num_clients in Cmin else Cmin['none']
                cmax = Cmax[num_clients] if num_clients in Cmax else Cmax['none']
                cnum = np.random.randint(cmin, cmax+1)
                classes = np.random.permutation(range(10))[:cnum]
                client_classes[i] = classes 
                for cls in classes: 
                    if cls in class_partition:
                        class_partition[cls] += 1
                    else:
                        class_partition[cls] = 1
            if len(class_partition) == 10: break
                
        # Calculate how to partition the dataset
        partition_endpoints = {}
        for label in labels:
            total_size = len(label_indices[label])

            # Partiton the samples from the same class to different lengths
            partitions = class_partition[label]
            partition_lengths = np.abs(np.random.normal(10, 3, size=partitions))

            # Scale the lengths so they add to the total length
            partition_lengths = partition_lengths / np.sum(partition_lengths) * total_size

            # Calculate the endpoints of each subrange
            endpoints = np.cumsum(partition_lengths)
            endpoints = np.array(endpoints, dtype=np.int32)
            endpoints[-1] = total_size
            partition_endpoints[label] = endpoints
        
        # Start dataset partition
        partition_pointer = {}
        for label in labels:
            partition_pointer[label] = 0
        client_datasets = []
        client_dataset_info = {}
        for i in range(num_clients):
            client_dataset_info[i] = {}
            sample_indices = []
            client_class = client_classes[i]
            for cls in client_class:
                start_idx = 0 if partition_pointer[cls] == 0 else partition_endpoints[cls][partition_pointer[cls]-1] # included
                end_idx = partition_endpoints[cls][partition_pointer[cls]] # excluded
                sample_indices.extend(label_indices[cls][start_idx:end_idx])
                partition_pointer[cls] += 1
                client_dataset_info[i][cls] = end_idx - start_idx # record the number for different classes
            client_datasets.append(sample_indices)

        train_data_input = []
        train_data_label = []
        for idx in client_datasets[client_idx]:
            train_data_input.append(data_raw[idx][0].tolist())
            train_data_label.append(data_raw[idx][1])
        return Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )

