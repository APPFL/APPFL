import os
import boto3 # Added for S3 interaction
import io    # Added for handling byte streams
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
            label_col: Name of the column containing the primary label.
            s3_client: Initialized boto3 S3 client.
            s3_bucket: Name of the S3 bucket.
            s3_prefix: Prefix (folder path) within the S3 bucket where images are located.
                       Example: 'cfp_images' if images are in s3://your-bucket/cfp_images/
            transform: torchvision transforms (augmentations) to apply
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col
        self.s3_client = boto3.client("s3") # Initialize S3 client
        self.s3_bucket = "sagemaker-ai-readi-tutorial-dataset"
        self.s3_prefix = 'cfp_images'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct the S3 object key using the prefix and the file path from the dataframe
        # Assumes file_path in df is relative to the s3_prefix
        # e.g., if s3_prefix='cfp_images' and row['file_path']='train/image1.png',
        # the key becomes 'cfp_images/train/image1.png'
        s3_key = os.path.join(self.s3_prefix, row["file_path"])
        label = row["label_idx"]

        try:
            # Get the image object from S3
            s3_object = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            # Read the image data from the object's body
            image_data = s3_object['Body'].read()
            # Load the image from the byte stream
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

        except Exception as e:
            print(f"Error loading image {s3_key} from S3 bucket {self.s3_bucket}: {e}")
            # Handle error appropriately: return None, a placeholder, or raise exception
            # Returning None might require collation function adjustments in DataLoader
            # For now, let's raise the exception to make failures explicit
            raise e

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
    s3_bucket: str = "sagemaker-ai-readi-tutorial-dataset",          # Added: S3 bucket name
    s3_prefix: str = "cfp_images", # Added: S3 prefix, defaults to 'cfp_images'
    partition_strategy: str = "iid",
    label_col: str = "device",
    partition_col: str = None,
    sampling_factor: int = None,
    aws_region: str = None,      # Optional: specify AWS region if needed by boto3
    **kwargs,
):
    """
    Loads and partitions the AI-READI dataset, streaming images from S3.

    It expects image data to be in s3://{s3_bucket}/{s3_prefix}/
    and the metadata file to be at s3://{s3_bucket}/{s3_prefix}/labels.tsv

    Args:
        num_clients: Total number of clients for federated learning.
        client_id: The ID of the current client (0 to num_clients-1).
        s3_bucket: The name of the S3 bucket containing the data.
        s3_prefix: The prefix (folder path) within the bucket. Defaults to 'cfp_images'.
                   Both images and labels.tsv are expected under this prefix.
        partition_strategy: How to partition the data ('iid', 'class_noniid', etc.).
        label_col: The column in labels.tsv to use as the primary classification label.
        partition_col: Column used for 'column_based' partitioning.
        sampling_factor: Fraction (0.0-1.0) of data to sample for faster dev/testing.
        aws_region: (Optional) AWS region for the S3 bucket.
        **kwargs: Additional arguments passed to partitioning functions.
    """
    # Initialize Boto3 S3 client
    # Boto3 automatically uses credentials from environment variables,
    # shared credential file (~/.aws/credentials), AWS config file (~/.aws/config),
    # or IAM role attached to EC2 instance/ECS task.
    s3_client = boto3.client("s3", region_name=aws_region)

    # Define the S3 key for the labels file
    tsv_key = os.path.join(s3_prefix, "labels.tsv") # e.g., "cfp_images/labels.tsv"

    try:
        # Get the labels.tsv file from S3
        print(f"Attempting to load labels from: s3://{s3_bucket}/{tsv_key}")
        obj = s3_client.get_object(Bucket=s3_bucket, Key=tsv_key)
        # Read the content and load into pandas DataFrame
        tsv_content = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(tsv_content), sep="\t")
        print(f"Successfully loaded labels.tsv from S3. Shape: {df.shape}")

    except Exception as e:
        print(f"Error loading {tsv_key} from S3 bucket {s3_bucket}: {e}")
        raise e

    # --- Rest of the preprocessing logic remains largely the same ---

    train_df = df[df["partition"] == "train"].copy()
    test_df = df[df["partition"] == "val"].copy()

    if train_df.empty or test_df.empty:
         print("Warning: Train or test dataframe is empty after filtering by partition.")
         # Handle this case appropriately, maybe raise an error or return empty datasets

    # Check if label_col exists
    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in the loaded labels.tsv data.")

    unique_classes = sorted(train_df[label_col].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    train_df["label_idx"] = train_df[label_col].map(class_to_idx)
    test_df["label_idx"] = test_df[label_col].map(class_to_idx)

    # down sampling for faster training time
    if sampling_factor is not None:
        print(f"Applying sampling factor: {sampling_factor}")
        train_df = train_df.sample(frac=sampling_factor, random_state=42).reset_index(
            drop=True
        )
        test_df = test_df.sample(frac=sampling_factor, random_state=42).reset_index(
            drop=True
        )
        print(f"Sampled train DF shape: {train_df.shape}, Sampled test DF shape: {test_df.shape}")


    train_transform = T.Compose(
        [
            T.Resize((224, 224)),  # Resize image to 224x224 pixels
            # T.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            # T.RandomRotation(degrees=15),  # Randomly rotate images
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

    # --- Partitioning logic remains the same ---
    print(f"Partitioning training data using strategy: {partition_strategy}")
    if partition_strategy == "iid":
        partitioned_datasets = iid_partition_df(train_df, num_clients)
    elif partition_strategy == "class_noniid":
        partitioned_datasets = class_noniid_partition_df(
            train_df,
            num_clients,
            label_col="label_idx",
            # Example Cmin/Cmax, adjust as needed
            # Cmin={cls: 1 for cls in class_to_idx.values()},
            # Cmax={cls: max(1, len(train_df[train_df['label_idx'] == cls]) // num_clients + 1) for cls in class_to_idx.values()},
            **kwargs,
        )
    elif partition_strategy == "dirichlet_noniid":
        partitioned_datasets = dirichlet_noniid_partition_df(
            train_df, num_clients, label_col="label_idx", **kwargs
        )
    elif partition_strategy == "column_based":
        if partition_col is None:
            raise ValueError("partition_col must be specified for 'column_based' strategy")
        if partition_col not in train_df.columns:
             raise ValueError(f"Partition column '{partition_col}' not found in the training data.")
        partitioned_datasets = column_based_partition_df(
            train_df,
            num_clients,
            label_col="label_idx",
            partition_col=partition_col,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    if client_id < 0 or client_id >= len(partitioned_datasets):
        raise ValueError(f"client_id {client_id} is out of range for {len(partitioned_datasets)} partitions.")

    client_train_df = partitioned_datasets[client_id]
    print(f"Client {client_id} train data shape: {client_train_df.shape}")
    if client_train_df.empty:
        print(f"Warning: Client {client_id} has an empty training dataset after partitioning.")

    # --- Create Dataset instances using the S3 info ---
    client_train_dataset = RetinopathyDataset(
        client_train_df,
        label_col=label_col,
        transform=train_transform
    )
    # The test dataset usually uses the full test set, not partitioned per client
    client_test_dataset = RetinopathyDataset(
        test_df,
        label_col=label_col,
        transform=val_transform
    )

    print(f"Created datasets for client {client_id}. Train size: {len(client_train_dataset)}, Test size: {len(client_test_dataset)}")

    return client_train_dataset, client_test_dataset
