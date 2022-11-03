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
    import torch

    
    class EcgDataset(Dataset):
        def __init__(self, hdf5_path, keys, labels, normalize_df=None, transform=None, target_transform=None):
            self.keys = keys
            self.labels = labels
            self.hdf5 = None
            self.hdf5_path = hdf5_path
            self.transform = transform
            self.target_transform = target_transform
            self.normalize_df = np.array(
                [[-139.941898,139.941898,279.883795],
                [-134.566014, 134.566014,269.132029],
                [-78.064679 , 78.064679 ,156.129359],
                [-129.421262,129.421262 ,258.842523],
                [-92.999893 ,92.999893  ,185.999786],
                [-88.909598 ,88.909598  ,177.819196],
                [-163.487644,163.487644 ,326.975288],
                [-202.913316,202.913316 ,405.826632],
                [-194.017951,194.017951 ,388.035902],
                [-222.348100,222.348100 ,444.696201],
                [-249.415011,249.415011 ,498.830022],
                [-216.065147,216.065147 ,432.130294]])

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
            return torch.tensor(ecg, dtype=torch.float), torch.tensor(label.astype(np.float), dtype=torch.float)

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
    meta_data = meta_data[meta_data['instance'] == 2]
    meta_data = meta_data[meta_data.bad == 0]
    meta_data.index = [os.path.join(os.path.split(a)[0],'data') for a in meta_data.index]

    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(meta_data.index, test_size=0.2, random_state=42)
    X_test, X_validation = train_test_split(X_test, test_size=0.5, random_state=42)
    if mode == "train":
        dataset = EcgDataset(h5py_path, X_train.values, meta_data.loc[X_train].ecg_age_years)
    elif mode == "val":
        dataset = EcgDataset(h5py_path, X_validation.values, meta_data.loc[X_validation].ecg_age_years)
    else:
        dataset = EcgDataset(h5py_path, X_test.values, meta_data.loc[X_test].ecg_age_years)
    return dataset