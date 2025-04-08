import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from appfl.misc.data import (
    iid_partition_df,
    class_noniid_partition_df,
    dirichlet_noniid_partition_df,
    column_based_partition_df,
)
import pandas as pd


class RetinopathyDataset(Dataset):
    def __init__(self, df, label_col, transform=None):
        """
        Args:
          df: a DataFrame with at least ['file_path', 'label_idx'] columns
          transform: torchvision transforms (augmentations) to apply
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = "cfp_images/" + row["file_path"]
        label = row["label_idx"]

        # load the image
        image = Image.open(img_path).convert("RGB")

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def label_counts(self):
        """Returns a dictionary of label_col and their counts."""
        return self.df[self.label_col].value_counts().to_dict()


def get_ai_readi(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    label_col: str = "device",
    partition_col: str = None,
    sampling_factor: int = None,
    **kwargs,
):
    """
    It expects data to be in working directory {WORKING_DIR}/cfp_images
    sampling_factor: 0.0 - 1.0
    Note: there should be a labels.tsv in the data directory mentioned above
    """
    tsv_path = os.getcwd() + "/cfp_images/labels.tsv"
    df = pd.read_csv(tsv_path, sep="\t")

    train_df = df[df["partition"] == "train"].copy()
    test_df = df[df["partition"] == "val"].copy()

    unique_classes = sorted(train_df[label_col].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    train_df["label_idx"] = train_df[label_col].map(class_to_idx)
    test_df["label_idx"] = test_df[label_col].map(class_to_idx)
    # down sampling for faster training time
    if sampling_factor is not None:
        train_df = train_df.sample(frac=sampling_factor, random_state=42).reset_index(
            drop=True
        )
        test_df = test_df.sample(frac=sampling_factor, random_state=42).reset_index(
            drop=True
        )

    train_transform = T.Compose(
        [
            T.Resize((224, 224)),  # Resize image to 224x224 pixels
            T.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            T.RandomRotation(degrees=15),  # Randomly rotate images
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize using ImageNet stats
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Run existing partitioning function
    if partition_strategy == "iid":
        partitioned_datasets = iid_partition_df(train_df, num_clients)
    elif partition_strategy == "class_noniid":
        partitioned_datasets = class_noniid_partition_df(
            train_df,
            num_clients,
            label_col="label_idx",
            Cmin={1: 4, 2: 2, 3: 2, "none": 1},
            Cmax={1: 4, 2: 2, 3: 3, "none": 4},
            **kwargs,
        )
    elif partition_strategy == "dirichlet_noniid":
        partitioned_datasets = dirichlet_noniid_partition_df(
            train_df, num_clients, label_col="label_idx", **kwargs
        )
    elif partition_strategy == "column_based":
        partitioned_datasets = column_based_partition_df(
            train_df,
            num_clients,
            label_col="label_idx",
            partition_col=partition_col,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    client_train_dataset = RetinopathyDataset(
        partitioned_datasets[client_id], label_col=label_col, transform=train_transform
    )
    client_test_dataset = RetinopathyDataset(
        test_df, label_col=label_col, transform=val_transform
    )

    return client_train_dataset, client_test_dataset
