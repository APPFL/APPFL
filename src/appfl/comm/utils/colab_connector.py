import os
import torch
from google.colab import drive


class GoogleColabConnector:
    def __init__(self, drive_path):
        self.drive_path = drive_path
        self._mount_drive()
        self._create_directory_if_missing()

    def _mount_drive(self):
        """Mount Google Drive if not already mounted."""
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")

    def _create_directory_if_missing(self):
        """Create the directory in Drive if it doesn't exist."""
        if not os.path.exists(self.drive_path):
            os.makedirs(self.drive_path)
            print(f"Created directory: {self.drive_path}")
        else:
            print(f"Directory already exists: {self.drive_path}")

    def upload(self, model, filename):
        """
        Save a PyTorch model to the Google Drive path.
        `model` can be a full model or a state_dict.
        """
        full_path = os.path.join(self.drive_path, filename)
        torch.save(model, full_path)
        print(f"PyTorch model saved to: {full_path}")
        return {"model_drive_path": full_path, "drive_path": self.drive_path}

    def __getstate__(self):
        """Serialize only the drive_path."""
        return {"drive_path": self.drive_path}

    def __setstate__(self, state):
        """Reinitialize the object after deserialization."""
        self.drive_path = state["drive_path"]
        self._mount_drive()
        self._create_directory_if_missing()
