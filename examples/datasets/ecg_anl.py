def get_data(
    cfg,
    client_idx: int,
    mode = "train"
):
    import os.path as osp
    from appfl.misc.data import Dataset
    import pandas as pd
    import os
    import blosc
    import h5py
    import numpy as np

    class EcgDataset(Dataset):
        def __init__(self, hdf5_path, keys, labels, transform=None, target_transform=None):
            self.keys = keys
            self.labels = labels
            self.hdf5 = None
            self.hdf5_path = hdf5_path
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, idx):
            if self.hdf5 == None:
                print(f'Opening file handle: {self.hdf5_path}')
                self.hdf5 = h5py.File(self.hdf5_path, 'r')
            
            index = self.keys[idx]
            ecg = self._uncompress_data(index)
            if ecg.shape[1] != 5000: # fix to not be fixed
                # Draw a random starting position
                slice_pos = np.random.randint(0, ecg.shape[1] - 5000)
                ecg = ecg[:,slice_pos:(slice_pos+5000)]

            label = self.labels[idx]
            if self.transform:
                ecg = self.transform(ecg)
            if self.target_transform:
                label = self.target_transform(label)
            return ecg, label

        def _uncompress_data(self, key, stored_dtype = np.int16):
            handle = self.hdf5[key]
            return np.frombuffer(
                blosc.decompress(handle[()]), dtype=stored_dtype
            ).reshape(handle.attrs["shape"]).astype(np.float32)

        # Pickle hackery
        def __getstate__(self):
            state = self.__dict__.copy()
            # Don't pickle hdf5
            del state["hdf5"]
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            # Add hdf5 back since it doesn't exist in the pickle
            self.hdf5 = None

    meta_path = cfg.clients[client_idx].data_pipeline.meta_path
    h5py_path = cfg.clients[client_idx].data_pipeline.h5py_path
    meta_data = pd.read_parquet(meta_path)
    meta_data['Age'] = meta_data['Age'].astype(np.float32)
    meta_data = meta_data[meta_data['n_observations'] >= 5000]
    meta_data = meta_data[~meta_data.Age.isna()]

    train_data = meta_data[meta_data['is_graded_train'] == True]
    test_data = meta_data[meta_data['is_graded_test'] == True]

    if mode == "train":
        dataset = EcgDataset(h5py_path, train_data.index.values, train_data.Age)
    else:
        dataset = EcgDataset(h5py_path, test_data.index.values, test_data.Age)
    return dataset