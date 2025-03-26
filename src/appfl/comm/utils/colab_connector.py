import os
import torch
from google.colab import drive
import time


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

    def load_model(self, filename, model_class=None, load_state_dict=False):
        """
        Load a PyTorch model or state_dict from the drive path.

        Args:
            filename (str): The name of the file in the drive directory.
            model_class (nn.Module, optional): The class of the model (required if loading state_dict).
            load_state_dict (bool): If True, loads state_dict into model_class.

        Returns:
            nn.Module: The loaded model.
        """
        # drive.flush_and_unmount()  # unmount
        # drive.mount("/content/drive")  # remount
        full_path = os.path.join(self.drive_path, filename)
        # if not os.path.exists(full_path):
        #     raise FileNotFoundError(f"No such file: {full_path}")

        timeout = 120
        start_time = time.time()

        while not os.path.exists(full_path):
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"File not found after {timeout} seconds: {full_path}"
                )
            time.sleep(5)

        if load_state_dict:
            if model_class is None:
                raise ValueError(
                    "model_class must be provided when loading state_dict."
                )
            model = model_class()
            state_dict = torch.load(full_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"State dict loaded into {model_class.__name__} from: {full_path}")
        else:
            model = torch.load(full_path, map_location="cpu")
            print(f"Full model loaded from: {full_path}")

        return model

    def __getstate__(self):
        """Serialize only the drive_path."""
        return {"drive_path": self.drive_path}

    def __setstate__(self, state):
        """Reinitialize the object after deserialization."""
        self.drive_path = state["drive_path"]
        self._mount_drive()
        self._create_directory_if_missing()
