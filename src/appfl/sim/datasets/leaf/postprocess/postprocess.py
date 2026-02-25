import os
import json
import logging

from appfl.sim.datasets.leaf.postprocess.sample import sample_clients
from appfl.sim.datasets.leaf.postprocess.filter import filter_clients
from appfl.sim.datasets.leaf.postprocess.split import split_datasets

logger = logging.getLogger(__name__)



def postprocess_leaf(
    dataset_name,
    root,
    seed,
    raw_data_fraction,
    min_samples_per_clients,
    test_size,
    logger=None,
):
    active_logger = logger or globals()["logger"]
    # check if raw data is prepared 
    active_logger.info(f'[LEAF-{dataset_name.upper()}] checking preprocessing artifacts.')
    if not os.path.exists(f'{root}/{dataset_name}/all_data'):
        err = f'[LEAF-{dataset_name.upper()}] preprocessing artifacts are missing under `{root}`.'
        raise AssertionError(err)
    active_logger.info(f'[LEAF-{dataset_name.upper()}] preprocessing artifacts are ready.')
    
    # create client datasets
    active_logger.info(f'[LEAF-{dataset_name.upper()}] sampling clients from raw pool.')
    if not os.path.exists(f'{root}/{dataset_name}/sampled_data'):
        os.makedirs(f'{root}/{dataset_name}/sampled_data')
        sample_clients(dataset_name, root, seed, raw_data_fraction)
    active_logger.info(f'[LEAF-{dataset_name.upper()}] sampled client pool is ready.')
    
    # remove clients with less than given `min_samples_per_clients`
    active_logger.info(f'[LEAF-{dataset_name.upper()}] filtering clients by minimum sample threshold.')
    if not os.path.exists(f'{root}/{dataset_name}/rem_clients_data') and (raw_data_fraction < 1.):
        os.makedirs(f'{root}/{dataset_name}/rem_clients_data')
        filter_clients(dataset_name, root, min_samples_per_clients)
    active_logger.info(f'[LEAF-{dataset_name.upper()}] client filtering complete.')
    
    # create train-test split
    active_logger.info(f'[LEAF-{dataset_name.upper()}] splitting clients into train/test sets.')
    if (not os.path.exists(f'{root}/{dataset_name}/train')) or (not os.path.exists(f'{root}/{dataset_name}/test')):
        if not os.path.exists(f'{root}/{dataset_name}/train'):
            os.makedirs(f'{root}/{dataset_name}/train')
        if not os.path.exists(f'{root}/{dataset_name}/test'):
            os.makedirs(f'{root}/{dataset_name}/test')    
        split_datasets(dataset_name, root, seed, test_size)
    active_logger.info(f'[LEAF-{dataset_name.upper()}] train/test split complete.')

    # get number of clients
    train_data = [file for file in os.listdir(os.path.join(root, dataset_name, 'train')) if file.endswith('.json')][0]
    num_clients = len(json.load(open(f'{root}/{dataset_name}/train/{train_data}', 'r'))['users'])
    return num_clients
