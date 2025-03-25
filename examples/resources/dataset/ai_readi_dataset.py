import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from appfl.misc.data import (
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)
import pandas as pd


class IndexLabelDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label_idx"]
        return idx, label  # index is the key here


# Defining a dataloader class, which returns image and its label
class RetinopathyDataset(Dataset):
    def __init__(self, df, indices, transform=None):
        """
        Args:
          df: a DataFrame with at least ['file_path', 'label_idx'] columns
          transform: torchvision transforms (augmentations) to apply
        """
        self.df = df.reset_index(drop=True)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        row = self.df.iloc[actual_idx]
        img_path = "cfp_images/" + row["file_path"]
        label = row["label_idx"]

        # load the image
        image = Image.open(img_path).convert("RGB")

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class RetinopathyTestDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
          df: a DataFrame with at least ['file_path', 'label_idx'] columns
          transform: torchvision transforms (augmentations) to apply
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

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


def get_ai_readi(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    print(os.getcwd())
    tsv_path = os.getcwd() + "/cfp_images/labels.tsv"
    df = pd.read_csv(tsv_path, sep="\t")

    train_df = df[df["partition"] == "train"].copy()
    test_df = df[df["partition"] == "test"].copy()

    unique_classes = sorted(train_df["device"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    train_df["label_idx"] = train_df["device"].map(class_to_idx)
    test_df["label_idx"] = test_df["device"].map(class_to_idx)

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

    # Use a lightweight dataset for partitioning (no image loading)
    index_label_dataset = IndexLabelDataset(train_df)
    print(index_label_dataset)

    # Run existing partitioning function
    if partition_strategy == "iid":
        partitioned_datasets = iid_partition(index_label_dataset, num_clients)
    elif partition_strategy == "class_noniid":
        partitioned_datasets = class_noniid_partition(
            index_label_dataset, num_clients, **kwargs
        )
    elif partition_strategy == "dirichlet_noniid":
        partitioned_datasets = dirichlet_noniid_partition(
            index_label_dataset, num_clients, **kwargs
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    client_partition = partitioned_datasets[client_id]
    partition_indices = [
        sample[0] for sample in client_partition
    ]  # sample is (index, label)
    partition_indices = [int(i.item()) for i in partition_indices]

    client_train_dataset = RetinopathyDataset(
        train_df, partition_indices, transform=train_transform
    )
    client_test_dataset = RetinopathyTestDataset(test_df, transform=val_transform)
    print(len(client_train_dataset))
    print(len(client_test_dataset))

    return client_train_dataset, client_test_dataset
