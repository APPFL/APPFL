import os
import torch
import torchvision
from mpi4py import MPI
from typing import Optional, List
from appfl.config import *
from appfl.misc.data import Dataset
from .utils.transform import test_transform, train_transform
from .utils.partition import iid_partition, class_noiid_partition, dirichlet_noiid_partition
from .utils.generate_readiness_report import generate_readiness_report

def get_mnist(
        comm: Optional[MPI.Comm], 
        num_clients: int,
        dr_metrics: Optional[List[str]]=None,
        partition: string = "iid",
        visualization: bool = True,
        output_dirname: string = "./outputs",
        cfg: Optional[Config]=None,
        **kwargs
):
    comm_rank = comm.Get_rank() if comm is not None else 0

    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    if comm_rank == 0:
        test_data_raw = torchvision.datasets.MNIST(dir, download=True, train=False, transform=test_transform("MNIST"))
    if comm is not None:
        comm.Barrier()
    if comm_rank > 0:
        test_data_raw = torchvision.datasets.MNIST(dir, download=False, train=False, transform=test_transform("MNIST"))

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))

    # Training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(dir, download=False, train=True, transform=train_transform("MNIST"))

    # Obtain the visualization output filename
    if visualization:
        dir = output_dirname
        if os.path.isdir(dir) == False:
            os.makedirs(dir, exist_ok=True)
        output_filename = f"MNIST_{num_clients}clients_{partition}_distribution"
        file_ext = ".pdf"
        filename = dir + "/%s%s" % (output_filename, file_ext)
        uniq = 1
        while os.path.exists(filename):
            filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
            uniq += 1
    else: filename = None

    # Partition the dataset
    if partition == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients, visualization=visualization and comm_rank==0, output=filename)
    elif partition == "class_noiid":
        train_datasets = class_noiid_partition(train_data_raw, num_clients, visualization=visualization and comm_rank==0, output=filename, **kwargs)
    elif partition == "dirichlet_noiid":
        train_datasets = dirichlet_noiid_partition(train_data_raw, num_clients, visualization=visualization and comm_rank==0, output=filename, **kwargs)
    
    # Obtain the data readines report output filename
    if dr_metrics is not None:
        drr_dir = output_dirname
        if os.path.isdir(drr_dir) == False:
            os.makedirs(drr_dir, exist_ok=True)
        drr_output_filename = f"MNIST_{num_clients}clients_{partition}_readiness_report"
        drr_file_ext = ".pdf"
        drr_filename = dir + "/%s%s" % (drr_output_filename, drr_file_ext)
        uniq = 1
        while os.path.exists(drr_filename):
            drr_filename = dir + "/%s_%d%s" % (drr_output_filename, uniq, drr_file_ext)
            uniq += 1
    else: drr_filename = None

    # data readiness report generation
    if dr_metrics is not None:
        generate_readiness_report(cfg,train_datasets=train_datasets,dr_metrics=dr_metrics,output_file=drr_filename,X=None,sens_attr=None)
    

    return train_datasets, test_dataset