import os
import torch
import torchvision
from mpi4py import MPI
from typing import Optional, List
from appfl.config import *
from appfl.misc.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from appfl.misc.data import Dataset
import torch
from .utils.partition import iid_partition, class_noiid_partition, dirichlet_noiid_partition
from .utils.generate_readiness_report import generate_readiness_report
from datasets.PreprocessedData.Credit_Preprocess import CreditData



def get_credit_data(
        comm: Optional[MPI.Comm], 
        num_clients: int,
        dr_metrics: Optional[List[str]]=None,
        partition: string = "iid",
        visualization: bool = True,
        output_dirname: string = "./outputs",
        sens_attr: Optional[List[str]] = None,
        cfg: Optional[Config]=None,
        **kwargs
):
    comm_rank = comm.Get_rank() if comm is not None else 0

    data_path = "datasets/PreprocessedData/Credit_Preprocess.csv"
    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)
    else:
        C = CreditData()
        C.preprocess()
        df = pd.read_csv(data_path)
    
    df_train = df.copy().drop(['Name','ID','Customer_ID','SSN'],axis=1)

    X  = df_train.drop(columns="Credit_Score")
    y  = df_train['Credit_Score']

    le      = LabelEncoder()
    y = le.fit_transform(y)
    pd.DataFrame(y).value_counts().sort_index()

    X_train_TOL_dummy = X['Type_of_Loan'].str.get_dummies(',').add_prefix('Type_of_Loan_')
    X_train_dummy = X.drop(['Type_of_Loan'],axis=1)
    X_train_dummy = pd.concat([X_train_dummy, X_train_TOL_dummy], axis=1)
    
    X_train_ohe = pd.get_dummies(X_train_dummy, columns=['Occupation', 'Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour'], prefix=['Occupation', 'Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour'])
    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_ohe), columns=X_train_ohe.columns)

    X_train_scaled = X_train_scaled.drop(['Month'],axis=1)

    x_numpy = X_train_scaled.to_numpy(dtype=np.float32)

    # Combine x and y arrays into a tuple or list
    x_y_data = list(zip(x_numpy, y))

    train_data_raw, test_data_raw = train_test_split(x_y_data, test_size=0.2, random_state=42)

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))

    # Obtain the visualization output filename
    if visualization:
        dir = output_dirname
        if os.path.isdir(dir) == False:
            os.makedirs(dir, exist_ok=True)
        output_filename = f"CREDITDATA_{num_clients}clients_{partition}_distribution"
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

    for i in range(num_clients):
        train_datasets[i].data_input = train_datasets[i].data_input.view(-1,1,6,9)
    
    test_dataset.data_input = test_dataset.data_input.view(-1,1,6,9)

    # Obtain the data readines report output filename
    if dr_metrics is not None:
        drr_dir = output_dirname
        if os.path.isdir(drr_dir) == False:
            os.makedirs(drr_dir, exist_ok=True)
        drr_output_filename = f"CREDITDATA_{num_clients}clients_{partition}_readiness_report"
        drr_file_ext = ".pdf"
        drr_filename = dir + "/%s%s" % (drr_output_filename, drr_file_ext)
        uniq = 1
        while os.path.exists(drr_filename):
            drr_filename = dir + "/%s_%d%s" % (drr_output_filename, uniq, drr_file_ext)
            uniq += 1
    else: drr_filename = None

    # data readiness report generation
    if dr_metrics is not None:
        generate_readiness_report(cfg, X,sens_attr,train_datasets,dr_metrics,drr_filename)

    return train_datasets, test_dataset



