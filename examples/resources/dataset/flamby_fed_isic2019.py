import os
import random
import io  # Added for in-memory file handling (BytesIO, StringIO)

import albumentations
import numpy as np
import pandas as pd
import torch
from PIL import Image

# boto3 is required for S3 access
try:
    import boto3
    from botocore.exceptions import (
        ClientError,
    )  # Import specific error for better handling
except ImportError:
    print(
        "WARNING: boto3 not found. S3 functionality will be unavailable. "
        "Install with: pip install boto3"
    )
    boto3 = None
    ClientError = None  # Define as None if boto3 import fails


# --- Helper function for joining S3 paths ---
def s3_join(*args):
    """Joins path components for S3, ensuring forward slashes."""
    # Filter out None or empty strings before joining
    return "/".join(s.strip("/") for s in args if s)


class Isic2019Raw(torch.utils.data.Dataset):
    """
    Pytorch dataset for Isic2019. Loads data and metadata from S3 or local disk.

    When using S3, both images and the metadata CSV are fetched from the specified bucket.
    When using local disk, uses the provided data_path structure.

    Attributes (when initialized):
    ----------
    metadata_df: pd.DataFrame
        DataFrame containing all loaded metadata (image, target, center, fold, fold2).
    image_identifiers: list[str]
        The list with the unique image names (e.g., 'ISIC_0000000') - potentially filtered by subclasses.
    targets: list[int]
        The list with classification labels corresponding to image_identifiers.
    centers: list[int]
        The list for datacenters corresponding to image_identifiers.
    X_dtype: torch.dtype
        The dtype of the X features output.
    y_dtype: torch.dtype
        The dtype of the y label output.
    augmentations: Callable
        Image transform operations from the albumentations library.
    use_s3: bool
        Flag indicating whether data is loaded from S3.
    s3_bucket: str
        Name of the S3 bucket (if use_s3 is True).
    s3_image_prefix: str
        Prefix (folder path) within S3 bucket where images are stored (if use_s3 is True).
    s3_client: boto3.client
        Initialized S3 client (if use_s3 is True).
    local_image_dir: str
        Path to the local directory containing images (if use_s3 is False).
    local_metadata_path: str
         Path to the local metadata file (if use_s3 is False).

    Parameters
    ----------
    X_dtype : torch.dtype, optional
        Defaults to torch.float32.
    y_dtype : torch.dtype, optional
        Defaults to torch.int64.
    augmentations : Callable, optional
        Albumentations augmentation pipeline. Defaults to None.
    data_path: str, optional
        Base path for local data (required if use_s3=False). Should contain the
        metadata file (e.g., 'train_test_split') and the image folder
        (e.g., 'ISIC_2019_Training_Input_preprocessed'). *Ignored if use_s3=True.*
    use_s3: bool, optional
        Set to True to load images and metadata from S3. Defaults to False.
    s3_bucket: str, optional
        The name of the S3 bucket. Required if use_s3 is True.
    s3_image_prefix: str, optional
        The prefix (folder path) in the S3 bucket where '.jpg' images are located
        (e.g., 'path/to/ISIC_2019_Training_Input_preprocessed'). Required if use_s3 is True.
    s3_metadata_key: str, optional
        The full S3 key (path within the bucket, e.g., 'path/to/metadata/train_test_split.csv')
        for the metadata CSV file. Required if use_s3 is True.
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        augmentations=None,
        data_path=None,  # Only used if use_s3 is False
        # S3 specific parameters
        use_s3: bool = True,
        s3_bucket: str = None,
        s3_image_prefix: str = None,
        s3_metadata_key: str = None,
    ):
        """
        Initializes the dataset from either S3 or local disk.
        """
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.augmentations = augmentations
        self.use_s3 = use_s3
        self.s3_bucket = s3_bucket
        self.s3_image_prefix = s3_image_prefix.strip("/") if s3_image_prefix else None
        self.s3_metadata_key = s3_metadata_key
        self.s3_client = None
        self.local_image_dir = None
        self.local_metadata_path = None
        self.metadata_df = None  # Will hold the full metadata

        if use_s3:
            # --- S3 Initialization ---
            if boto3 is None:
                raise ImportError(
                    "boto3 is required for S3 support. Please install it: pip install boto3"
                )
            if not s3_bucket or not self.s3_image_prefix or not s3_metadata_key:
                raise ValueError(
                    "s3_bucket, s3_image_prefix, and s3_metadata_key must be provided when use_s3 is True."
                )
            print(f"Initializing S3 client for bucket '{s3_bucket}'")
            self.s3_client = boto3.client("s3")

            # --- Load Metadata from S3 ---
            print(f"Loading metadata from S3: s3://{s3_bucket}/{s3_metadata_key}")
            try:
                s3_response = self.s3_client.get_object(
                    Bucket=self.s3_bucket, Key=self.s3_metadata_key
                )
                csv_content = s3_response["Body"].read()
                # Read CSV from bytes using io.BytesIO
                self.metadata_df = pd.read_csv(io.BytesIO(csv_content))
                print("Metadata loaded successfully from S3.")

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == "NoSuchKey":
                    raise FileNotFoundError(
                        f"Metadata file not found in S3 at key: {self.s3_metadata_key} in bucket {self.s3_bucket}"
                    )
                elif error_code == "AccessDenied":
                    raise PermissionError(
                        f"Access denied when trying to read S3 key: {self.s3_metadata_key}"
                    )
                else:
                    raise OSError(
                        f"S3 Error reading metadata from s3://{s3_bucket}/{s3_metadata_key}: {e}"
                    )
            except Exception as e:
                raise OSError(f"Error parsing metadata CSV from S3: {e}")

        else:
            # --- Local Disk Initialization ---
            print("Using local file system.")
            if not data_path or not os.path.isdir(data_path):
                raise ValueError(
                    f"A valid local 'data_path' must be provided when use_s3 is False. Got: {data_path}"
                )

            # Define expected local paths based on data_path
            # Assuming metadata file is named 'train_test_split' directly under data_path
            # And images are in 'ISIC_2019_Training_Input_preprocessed' under data_path
            self.local_metadata_path = os.path.join(data_path, "train_test_split")
            self.local_image_dir = os.path.join(
                data_path, "ISIC_2019_Training_Input_preprocessed"
            )

            print(f"Expecting metadata at: {self.local_metadata_path}")
            print(f"Expecting images in: {self.local_image_dir}")

            if not os.path.exists(self.local_metadata_path):
                raise FileNotFoundError(
                    f"Local metadata file not found at: {self.local_metadata_path}"
                )
            if not os.path.isdir(self.local_image_dir):
                raise NotADirectoryError(
                    f"Local image directory not found at: {self.local_image_dir}"
                )

            # --- Load Metadata from Local Disk ---
            try:
                self.metadata_df = pd.read_csv(self.local_metadata_path)
                print(f"Metadata loaded successfully from {self.local_metadata_path}.")
            except Exception as e:
                raise OSError(
                    f"Error reading local metadata file {self.local_metadata_path}: {e}"
                )

        # --- Common Initialization (after metadata loaded) ---
        # Validate required columns exist
        required_cols = ["image", "target", "center"]  # Base required columns
        # Add fold/fold2 check later if needed, FedIsic needs them
        if not all(col in self.metadata_df.columns for col in required_cols):
            raise ValueError(
                f"Metadata CSV must contain columns: {required_cols}. "
                f"Found: {self.metadata_df.columns.tolist()}"
            )

        # Store initial lists (subclasses might filter these)
        # IMPORTANT: Store only identifiers (names), not full paths
        self.image_identifiers = self.metadata_df["image"].tolist()
        self.targets = self.metadata_df[
            "target"
        ].tolist()  # Use .tolist() for consistency
        self.centers = self.metadata_df["center"].tolist()

        # Remove the old self.dic and self.image_paths list construction
        # self.dic = {...} # Removed
        # self.image_paths = [...] # Removed

    def __len__(self):
        # Length depends on the potentially filtered identifiers list
        return len(self.image_identifiers)

    def __getitem__(self, idx):
        # Use the potentially filtered identifier list
        if idx >= len(self.image_identifiers):
            raise IndexError(
                f"Index {idx} out of bounds for dataset length {len(self.image_identifiers)}"
            )

        image_name = self.image_identifiers[idx]
        # Get target and center from the potentially filtered lists
        target = self.targets[idx]
        # center = self.centers[idx] # Can be accessed if needed

        image_filename = f"{image_name}.jpg"  # Assuming .jpg extension

        try:
            if self.use_s3:
                # --- Load image from S3 ---
                s3_key = s3_join(self.s3_image_prefix, image_filename)
                try:
                    s3_response = self.s3_client.get_object(
                        Bucket=self.s3_bucket, Key=s3_key
                    )
                    image_data = s3_response["Body"].read()
                    # Open image from in-memory bytes
                    image = Image.open(io.BytesIO(image_data)).convert(
                        "RGB"
                    )  # Ensure RGB
                except ClientError as e:
                    # Handle S3 errors specifically for the image
                    error_code = e.response.get("Error", {}).get("Code")
                    if error_code == "NoSuchKey":
                        # Provide more context in error message
                        raise FileNotFoundError(
                            f"Image not found in S3: s3://{self.s3_bucket}/{s3_key} "
                            f"(derived from identifier '{image_name}' at index {idx})"
                        )
                    else:
                        raise OSError(f"S3 Error loading image {s3_key}: {e}")

            else:
                # --- Load image from Local Disk ---
                image_path = os.path.join(self.local_image_dir, image_filename)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(
                        f"Local image file not found: {image_path} "
                        f"(derived from identifier '{image_name}' at index {idx})"
                    )
                image = Image.open(image_path).convert("RGB")  # Ensure RGB

            # Convert PIL Image to NumPy array
            image = np.array(image)

            # Apply Albumentations augmentations
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]

            # Transpose and convert type (C, H, W) format for PyTorch
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

            # Convert to PyTorch tensors
            image_tensor = torch.tensor(image, dtype=self.X_dtype)
            target_tensor = torch.tensor(target, dtype=self.y_dtype)

            return image_tensor, target_tensor

        except FileNotFoundError as e:
            print(f"Error in __getitem__ (idx={idx}): {e}")
            # Consider returning None or placeholder, depends on DataLoader collation
            # Returning dummy data to avoid crashing DataLoader worker
            # Try to get expected size from augmentations if possible (might fail if None)
            sz = 200  # Default size
            try:
                if self.augmentations:
                    # Check for common size attributes in crop/resize transforms
                    if hasattr(self.augmentations, "height"):
                        sz = self.augmentations.height
                    elif any(
                        hasattr(t, "height")
                        for t in getattr(self.augmentations, "transforms", [])
                    ):
                        sz = next(
                            t.height
                            for t in self.augmentations.transforms
                            if hasattr(t, "height")
                        )
            except Exception:
                pass  # Stick with default size if check fails
            dummy_image = torch.zeros((3, sz, sz), dtype=self.X_dtype)
            dummy_target = torch.tensor(-1, dtype=self.y_dtype)  # Invalid target
            return dummy_image, dummy_target
        except Exception as e:
            # Catch other potential errors (PIL errors, augmentation errors, etc.)
            print(
                f"Error processing item idx={idx}, image_name='{image_name}': {type(e).__name__} - {e}"
            )
            sz = 200
            try:  # Reuse size detection logic
                if self.augmentations:
                    if hasattr(self.augmentations, "height"):
                        sz = self.augmentations.height
                    elif any(
                        hasattr(t, "height")
                        for t in getattr(self.augmentations, "transforms", [])
                    ):
                        sz = next(
                            t.height
                            for t in self.augmentations.transforms
                            if hasattr(t, "height")
                        )
            except Exception:
                pass
            dummy_image = torch.zeros((3, sz, sz), dtype=self.X_dtype)
            dummy_target = torch.tensor(-1, dtype=self.y_dtype)
            return dummy_image, dummy_target


class FedIsic2019(Isic2019Raw):
    """
    Federated Isic2019 dataset, inheriting S3/local capabilities from Isic2019Raw.

    Filters the base dataset based on center, train/test split, and pooling options
    using the metadata loaded by the parent class. Requires 'fold' and 'fold2' columns
    in the metadata.
    """

    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64,
        # --- Parameters passed to Isic2019Raw ---
        data_path: str = None,  # Only used if use_s3 is False
        use_s3: bool = False,
        s3_bucket: str = None,
        s3_image_prefix: str = None,
        s3_metadata_key: str = None,
    ):
        """Initializes the federated dataset, applying filters."""
        sz = 200  # Standard size for cropping/padding
        if train:
            # Define train augmentations
            augmentations = albumentations.Compose(
                [
                    albumentations.RandomScale(
                        scale_limit=0.07, p=0.5
                    ),  # Added p for probability
                    albumentations.Rotate(limit=50, p=0.5),  # Added p
                    albumentations.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.1, p=0.5
                    ),  # Added p
                    albumentations.HorizontalFlip(p=0.5),  # Corrected Flip call
                    albumentations.Affine(shear=0.1, p=0.5),  # Added p
                    # RandomCrop *before* Normalize if size varies
                    albumentations.RandomCrop(height=sz, width=sz),
                    albumentations.CoarseDropout(
                        max_holes=random.randint(
                            1, 8
                        ),  # Keep random hole count per image
                        max_height=16,
                        max_width=16,
                        min_holes=1,
                        min_height=8,
                        min_width=8,  # Added min values for example
                        p=0.5,  # Probability of applying dropout
                    ),
                    albumentations.Normalize(always_apply=True),
                ]
            )
        else:  # Test augmentations
            augmentations = albumentations.Compose(
                [
                    # Ensure test images are also consistently sized
                    albumentations.CenterCrop(height=sz, width=sz),
                    albumentations.Normalize(always_apply=True),
                ]
            )

        # --- Initialize the base class (Isic2019Raw) ---
        # This will load the full metadata_df from S3 or local disk
        # Pass all relevant parameters up
        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            augmentations=augmentations,
            data_path=data_path,
            use_s3=use_s3,
            s3_bucket=s3_bucket,
            s3_image_prefix=s3_image_prefix,
            s3_metadata_key=s3_metadata_key,
        )

        # --- Filtering logic using the parent's loaded metadata_df ---
        self.center = center
        self.train_test = "train" if train else "test"
        self.pooled = pooled

        if self.metadata_df is None:
            raise RuntimeError(
                "Base class (Isic2019Raw) did not load metadata correctly."
            )

        # Check for required filtering columns ('fold', 'fold2')
        required_filter_cols = ["fold", "fold2"]
        if not all(col in self.metadata_df.columns for col in required_filter_cols):
            raise ValueError(
                f"Metadata CSV must contain columns: {required_filter_cols} for filtering. "
                f"Found: {self.metadata_df.columns.tolist()}"
            )

        df_to_filter = self.metadata_df  # Use the full dataframe loaded by parent

        # Apply filters based on pooled/center settings
        if self.pooled:
            # Filter by 'train' or 'test' using the 'fold' column
            # Using @self.train_test allows variable injection into the query string safely
            df_filtered = df_to_filter.query("fold == @self.train_test").reset_index(
                drop=True
            )
            print(
                f"Filtering for pooled '{self.train_test}' split using 'fold' column."
            )
        else:
            # Filter by specific center and train/test split using 'fold2' column
            if not (0 <= center < 6):  # Assuming 6 centers (0-5)
                raise ValueError(
                    f"Invalid center specified: {center}. Must be between 0 and 5."
                )
            # Construct the key like 'train_0', 'test_3', etc.
            split_key = f"{self.train_test}_{self.center}"
            # Use @split_key for safe variable injection
            df_filtered = df_to_filter.query("fold2 == @split_key").reset_index(
                drop=True
            )
            print(
                f"Filtering for center {center}, '{self.train_test}' split using 'fold2' column (key='{split_key}')."
            )

        # --- Update the dataset attributes with the filtered data ---
        # Replace the lists inherited from the parent with the filtered versions
        self.image_identifiers = df_filtered["image"].tolist()
        self.targets = df_filtered["target"].tolist()
        self.centers = df_filtered["center"].tolist()
        # Optionally keep the filtered dataframe itself if needed elsewhere
        # self.filtered_metadata_df = df_filtered

        print(
            f"FedIsic2019 Initialized: Center={center if not pooled else 'Pooled'}, "
            f"Split='{self.train_test}', use_s3={use_s3}. "
            f"Filtered dataset size: {len(self.image_identifiers)}"
        )
        if len(self.image_identifiers) == 0:
            print(
                "WARNING: Dataset is empty after filtering. Check metadata and filter criteria."
            )


def get_flamby_fed_isic2019(
    client_id: int,
    data_path=None,  # Only used if use_s3 is False
    pooled: bool = False,
    # S3 specific parameters
    use_s3: bool = True,
    s3_bucket: str = None,
    s3_image_prefix: str = None,
    s3_metadata_key: str = None,
    **kwargs,
):
    train_dataset = FedIsic2019(
        center=client_id,
        train=True,
        pooled=pooled,
        data_path=data_path,  # Will be None if use_s3 is True
        use_s3=use_s3,
        s3_bucket=s3_bucket,
        s3_image_prefix=s3_image_prefix,
        s3_metadata_key=s3_metadata_key,
    )

    test_dataset = FedIsic2019(
        center=client_id,
        train=False,
        pooled=pooled,
        data_path=data_path,  # Will be None if use_s3 is True
        use_s3=use_s3,
        s3_bucket=s3_bucket,
        s3_image_prefix=s3_image_prefix,
        s3_metadata_key=s3_metadata_key,
    )

    return train_dataset, test_dataset
