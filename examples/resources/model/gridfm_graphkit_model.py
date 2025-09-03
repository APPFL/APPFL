from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask

import yaml

def get_gridfm_graphkit_model():
    config_path = "./resources/configs/grid/gridfm_graphkit.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_args = NestedNamespace(**config_dict)

    data_module = LitGridDataModule(config_args, "data")
    data_module.setup("train")

    model = FeatureReconstructionTask(
        config_args, data_module.node_normalizers, data_module.edge_normalizers
    )

    return model