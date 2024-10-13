import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from appfl.misc.data import Dataset, iid_partition, dirichlet_noniid_partition, class_noniid_partition

def get_adult(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    test_size: float = 0.2,
    **kwargs
):
    """
    Return the Adult Income dataset for a given client, with optional class balancing.
    :param num_clients: total number of clients
    :param client_id: the client id
    :param partition_strategy: strategy to partition the data ("iid" or "dirichlet_noniid")
    :param test_size: proportion of the dataset to include in the test split
    :param balance_classes: whether to balance the classes in the dataset (default is True)
    """
    # Get the download directory for dataset
    dir = os.path.join(os.getcwd(), "datasets", "RawData")
    os.makedirs(dir, exist_ok=True)

    # Download the data if not already available
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dataset_path = os.path.join(dir, "adult.csv")
    if not os.path.exists(dataset_path):
        pd.read_csv(url, header=None).to_csv(dataset_path, index=False)

    # Load and preprocess the data
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv(dataset_path, names=column_names, skipinitialspace=True)

    # Clean the income column
    data['income'] = data['income'].str.strip().replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    
    # Remove any rows with unexpected income values
    valid_incomes = ['<=50K', '>50K']
    data = data[data['income'].isin(valid_incomes)]

    # Encode categorical variables
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']):
        if column != 'income':
            data[column] = le.fit_transform(data[column])

    # Handle income classes
    income_le = LabelEncoder()
    data['income'] = income_le.fit_transform(data['income'])

    # Split features and target
    X = data.drop('income', axis=1)
    y = data['income']

    # Handle sensitive attributes if specified
    sensitive_attribute = kwargs.get("sensitive_attribute", None)
    if sensitive_attribute is not None:
        if sensitive_attribute not in data.columns:
            raise ValueError(f"Invalid sensitive attribute: {sensitive_attribute}")
        sensitive_attribute_data = data[sensitive_attribute].values

    # Balance the classes manually if balance_classes is True
    if kwargs.get("balance_classes") is True:
        class_0 = data[data['income'] == 0]  # Class 0: income <= 50K
        class_1 = data[data['income'] == 1]  # Class 1: income > 50K
        
        # Oversample the minority class
        if len(class_0) > len(class_1):
            class_1_oversampled = class_1.sample(len(class_0), replace=True, random_state=42)
            balanced_data = pd.concat([class_0, class_1_oversampled], axis=0)
        else:
            class_0_oversampled = class_0.sample(len(class_1), replace=True, random_state=42)
            balanced_data = pd.concat([class_1, class_0_oversampled], axis=0)

        # Shuffle the dataset
        balanced_data = balanced_data.sample(frac=1, random_state=42)

        # After balancing, split features and target
        X = balanced_data.drop('income', axis=1)
        y = balanced_data['income']

        # Handle sensitive attribute balancing (if applicable)
        if sensitive_attribute:
            sensitive_attribute_data = balanced_data[sensitive_attribute].values

    # Scale the features AFTER balancing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    if sensitive_attribute:
        sens_train, sens_test = train_test_split(sensitive_attribute_data, test_size=test_size, random_state=42, stratify=y)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    if sensitive_attribute:
        sens_train_tensor = torch.LongTensor(sens_train)
        sens_test_tensor = torch.LongTensor(sens_test)

    # Create Dataset object with or without sensitive attribute
    if sensitive_attribute:
        train_data_raw = Dataset(X_train_tensor, y_train_tensor, sens_train_tensor)
        test_dataset = Dataset(X_test_tensor, y_test_tensor, sens_test_tensor)
    else:
        train_data_raw = Dataset(X_train_tensor, y_train_tensor)
        test_dataset = Dataset(X_test_tensor, y_test_tensor)
    
    # Partition the dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(train_data_raw, num_clients, **kwargs)
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")
    
    return train_datasets[client_id], test_dataset