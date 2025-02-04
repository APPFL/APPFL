from appfl.misc.data_readiness import *
from torch.utils.data import Subset
import numpy as np
try:
    from flamby.datasets.fed_tcga_brca import FedTcgaBrca
    from flamby.datasets.fed_heart_disease import FedHeartDisease
    from flamby.datasets.fed_ixi import FedIXITiny
    from flamby.datasets.fed_isic2019 import FedIsic2019
    from flamby.datasets.fed_kits19 import FedKits19
except:
    pass

def get_flamby(
    dataset: str, 
    num_clients: int,
    client_id: int,
    **kwargs,
):
    if dataset == 'TcgaBrca':
        assert num_clients <= 6, "TcgaBrca dataset can support at most six clients"
        test_dataset =  FedTcgaBrca(train=False, pooled=True)
        train_dataset = FedTcgaBrca(train=True, center=client_id, pooled=False)     
        return train_dataset, test_dataset
    elif dataset == 'HeartDisease':
        assert num_clients <= 4, "Heart disease dataset can support at most four clients"
        test_dataset =  FedHeartDisease(train=False, pooled=True)
        train_dataset = FedHeartDisease(train=True, center=client_id, pooled=False)
        return train_dataset, test_dataset
    elif dataset == 'IXI':
        assert num_clients <= 3, "IXI dataset can support at most three clients"
        
        test_dataset = FedIXITiny(train=False, pooled=True)
        train_dataset = FedIXITiny(train=True, center=client_id, pooled=False) 

        # Check if the dataset has more than 59 samples
        # if len(train_dataset) > 59:
        #     # Randomly select 59 indices from the dataset
        #     indices = np.random.choice(len(train_dataset), 59, replace=False)
        #     # Use Subset to create a smaller dataset with 59 samples
        #     train_dataset = Subset(train_dataset, indices)

        # # Add equal proportion of noise to each client's dataset
        if client_id == 0:
        
            train_dataset = add_noise_to_subset(train_dataset, scale=2, fraction=0.95)

        return train_dataset, test_dataset
    elif dataset == 'ISIC2019':
        assert num_clients <= 6, "ISIC 2019 dataset can support at most six clients"
        test_dataset = FedIsic2019(train=False, pooled=True)
        train_dataset = FedIsic2019(train=True, center=client_id, pooled=False)
        return train_dataset, test_dataset
    elif dataset == 'Kits19':
        assert num_clients <= 6, "Kits19 dataset can support at most six clients"
        test_dataset = FedKits19(train=False, pooled=True)
        train_dataset = FedKits19(train=True, center=client_id, pooled=False) 
        return train_dataset, test_dataset
    else:
        raise NotImplementedError    


