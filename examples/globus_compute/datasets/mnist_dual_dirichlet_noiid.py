def get_data(cfg, client_idx: int, mode='train'):
    import torch
    import numpy as np
    import torchvision
    import os.path as osp
    from appfl.misc.data import Dataset
    from torchvision.transforms import ToTensor

    # Prepare local dataset directory
    data_dir = cfg.clients[client_idx].data_dir
    local_dir = osp.join(data_dir,"RawData")
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
        # Partition the dataset using dual-Dirichlet Non-IID
        np.random.seed(42)
        # The following code partition the dataset using the dual-Dirichlet partition strategy so that each client only has a subset of the labels
        # The code is just a reform of the original code in examples/dataloader/utils/partition.py
        # Split the dataset by label
        labels = []
        label_indices = {}
        for idx, (_, label) in enumerate(data_raw):
            if label not in label_indices:
                label_indices[label] = []
                labels.append(label)
            label_indices[label].append(idx)
        labels.sort()

        # Shuffle the indices for different label
        for label in labels:
            np.random.shuffle(label_indices[label])

        num_clients = cfg.num_clients
        alpha1 = num_clients
        alpha2 = 0.5

        p1 = [1 / num_clients for _ in range(num_clients)]      # prior distribution for each client's number of elements
        p2 = [len(label_indices[label]) for label in labels]
        p2 = [p / sum(p2) for p in p2]                          # prior distribution for each class's number of elements

        q1 = [alpha1 * i for i in p1]
        q2 = [alpha2 * i for i in p2]

        weights = np.random.dirichlet(q1) # the total number of elements for each client
        individuals = np.random.dirichlet(q2, num_clients) # the number of elements from each class for each client

        classes = [len(label_indices[label]) for label in labels]

        normalized_portions = np.zeros(individuals.shape)
        for i in range(num_clients):
            for j in range(len(classes)):
                normalized_portions[i][j] = weights[i] * individuals[i][j] / np.dot(weights, individuals.transpose()[j])

        res = np.multiply(np.array([classes] * num_clients), normalized_portions).transpose()

        for i in range(len(classes)):
            total = 0
            for j in range(num_clients - 1):
                res[i][j] = int(res[i][j])
                total += res[i][j]
            res[i][num_clients - 1] = classes[i] - total


        # number of elements from each class for each client
        num_elements = np.array(res.transpose(), dtype=np.int32)
        sum_elements = np.cumsum(num_elements, axis=0)

        train_data_input = []
        train_data_label = []
        for j, label in enumerate(labels):
            start = 0 if client_idx == 0 else sum_elements[client_idx-1][j]
            end = sum_elements[client_idx][j]
            for idx in label_indices[label][start:end]:
                train_data_input.append(data_raw[idx][0].tolist())
                train_data_label.append(data_raw[idx][1])
        return Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
