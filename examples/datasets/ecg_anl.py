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
            self.normalize_df = np.array(
                [[-813.318497,813.318497,1626.636994],
                [-927.557905,927.557905,1855.115811],
                [-616.430969,616.430969,1232.861938],
                [-779.541208,779.541208,1559.082417],
                [-598.479875,598.479875,1196.959749],
                [-703.899302,703.899302,1407.798603],
                [-1011.058909,1011.058909,2022.117818],
                [-1519.952699,1519.952699,3039.905399],
                [-1354.500198,1354.500198,2709.000397],
                [-1705.832194,1705.832194,3411.664388],
                [-1745.244320,1745.244320,3490.488641],
                [-1486.945966,1486.945966,2973.891932]])

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
            if self.normalize_df is not None:
                # Assumed min,max,range as columns in pd.DataFrame
                ecg = np.stack([ 2 * ( (ecg[i] - self.normalize_df[i,0]) / (self.normalize_df[i,2]) ) - 1 for i in range(ecg.shape[0]) ])
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