import torch
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from .drmetrics.imbalance_degree import imbalance_degree
from .drmetrics.plots import plot_class_distribution_subplots_to_pdf,plot_kde_to_pdf



def generate_readiness_report(train_datasets, dr_metrics,output_filename):
    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10,"Data Characteristics Report",align="C")
    pdf.ln()

    # print("------- Data Readiness Report - Begin --------")

    num_clients = len(train_datasets)
    # aggregate_dr_scores = [0] * num_clients  # Initialize scores for each client

    #text information
    if "ci" in dr_metrics:
        pdf.cell(200, 10, "Class Imbalance Degree:", ln=1)
        pdf.multi_cell(200, 10, "Class imbalance degree indicates the extent to which the class distribution is imbalanced. A higher value signifies a greater imbalance.")

        # print("Class Imbalance Degree:")
        labels_list = []
        for i, dataset in enumerate(train_datasets):
            labels = dataset.data_label.tolist()
            labels_list.append(labels)
            imb_deg = imbalance_degree(labels)
            # print(f"    Client {i + 1}: {round(imb_deg, 2)}")
            pdf.cell(200, 10, f"    Client {i + 1}: {round(imb_deg, 2)}", ln=1)
            # aggregate_dr_scores[i] += (1 - imb_deg)


    if "comp" in dr_metrics:
        pdf.cell(200, 10, "Completeness:", ln=1)
        pdf.multi_cell(200, 10, "Completeness indicates the proportion of non-missing (non-NaN) values in the dataset. A higher value signifies more complete data.")

        # print("Completeness:")
        for i, dataset in enumerate(train_datasets):
            # Access the data from the current dataset
            data = dataset.data_input
            # Calculate the proportion of non-NaN values
            num_non_nan = torch.sum(~torch.isnan(data))
            total_elements = data.numel()
            completeness = num_non_nan.item() / total_elements
            # print(f"    Client {i + 1}: {round(completeness, 2)}")
            pdf.cell(200, 10, f"    Client {i + 1}: {round(completeness, 2)}", ln=1)
        
            # aggregate_dr_scores[i] += completeness

    if "ss" in dr_metrics:
        pdf.cell(200, 10, "Sample Size:", ln=1)
        pdf.multi_cell(200, 10, "Sample size indicates the number of samples in the dataset for each client.")

        # print("Sample Size:")

        for i,dataset in  enumerate(train_datasets):
            sample_size = len(dataset.data_label.tolist())
            # print(f"    Client {i + 1}: {round(sample_size)}")
            pdf.cell(200, 10, f"    Client {i + 1}: {round(sample_size)}", ln=1)

    if "dim" in dr_metrics:
        pdf.cell(200, 10, "Dimensions", ln=1)
        pdf.multi_cell(200, 10, "Dimensions represent the shape of the data array, indicating the number of samples and features.")

        # print("Dimensions")
        for i, dataset in enumerate(train_datasets):
            data = dataset.data_input
            # print(f"    Client {i + 1}: Shape = {tuple(data.shape)}")
            pdf.cell(200, 10, f"    Client {i + 1}: Shape = {tuple(data.shape)}", ln=1)
    
    if "range" in dr_metrics:
        pdf.cell(200, 10, "Value Range", ln=1)
        pdf.multi_cell(200, 10, "Value range indicates the minimum and maximum values present in the dataset.")

        # print("Value Range")
        for i, dataset in enumerate(train_datasets):
            data = dataset.data_input
            min_val = torch.min(data)
            max_val = torch.max(data)
            # print(f"    Client {i + 1}: Min Value = {min_val.item()}, Max Value = {max_val.item()}")
            pdf.cell(200, 10, f"    Client {i + 1}: Min Value = {min_val.item()}, Max Value = {max_val.item()}", ln=1)
    
    
    if "sparsity" in dr_metrics:
        pdf.cell(200, 10, "Sparsity:", ln=1)
        pdf.multi_cell(200, 10, "Sparsity indicates the proportion of zero values in the dataset. A higher value signifies more sparse data.")

        # print("Sparsity")
        for i, dataset in enumerate(train_datasets):
            total_elements = data.numel()
            num_zeros = torch.sum(data == 0)
            sparsity = num_zeros.item() / total_elements
            # print(f"    Client {i + 1}: Sparsity = {round(sparsity, 4)}")
            pdf.cell(200, 10, f"    Client {i + 1}: Sparsity = {round(sparsity, 4)}", ln=1)

    if "variance" in dr_metrics:
        pdf.cell(200, 10, "Variance:", ln=1)
        pdf.multi_cell(200, 10, "Variance measures the spread of the data values. A higher value indicates more variability.")

        for i, dataset in enumerate(train_datasets):
            data = dataset.data_input
            variance = torch.var(data)
            pdf.cell(200, 10, f"    Client {i + 1}: Variance = {round(variance.item(), 2)}", ln=1)

    if "skewness" in dr_metrics:
        pdf.cell(200, 10, "Skewness:", ln=1)
        pdf.multi_cell(200, 10, "Skewness measures the asymmetry of the data distribution. Positive values indicate right skew, while negative values indicate left skew.")

        for i, dataset in enumerate(train_datasets):
            data = dataset.data_input.numpy().flatten()
            mean = np.mean(data)
            std_dev = np.std(data)
            skewness = np.mean(((data - mean) / std_dev) ** 3)
            rounded_skewness = round(skewness,2)
            pdf.cell(200, 10, f"    Client {i + 1}: Skewness = {rounded_skewness}", ln=1)

    if "entropy" in dr_metrics:
        pdf.cell(200, 10, "Entropy:", ln=1)
        pdf.multi_cell(200, 10, "Entropy measures the uncertainty or randomness in the dataset. Higher values indicate more disorder.")

        for i, dataset in enumerate(train_datasets):
            data = dataset.data_input.numpy().flatten()
            entropy = -np.sum(data * np.log2(data + 1e-9)) / len(data)  # Adding a small constant to avoid log(0)
            rounded_entropy = round(entropy,2)
            pdf.cell(200, 10, f"    Client {i + 1}: Entropy = {rounded_entropy}", ln=1)

    # plotting first
    if "kde" in dr_metrics:
        data_list = []
        for i, dataset in enumerate(train_datasets):
            data = dataset.data_input
            data_list.append(data)
        plot_kde_to_pdf(data_list, pdf, len(train_datasets))
        
    
    if "ci" in dr_metrics:
        labels_list = []
        for i, dataset in enumerate(train_datasets):
            labels = dataset.data_label.tolist()
            labels_list.append(labels)
        plot_class_distribution_subplots_to_pdf(labels_list, pdf, len(train_datasets))

    # if "rr" in dr_metrics:
    #     print("Representation Rate:")
    #     for i, dataset in enumerate(train_datasets):
    #         gender_list = dataset.data_gender.tolist()
    #         count_1s = gender_list.count(1)
    #         count_0s = gender_list.count(0)
    #         total_samples = len(gender_list)
    #         representation_rate_1s = count_1s / total_samples
    #         representation_rate_0s = count_0s / total_samples
    #         print(f"  Client {i + 1}: Representation Rate of Males: {round(representation_rate_1s, 4)}, Representation Rate of Females: {round(representation_rate_0s, 4)}")

    num_metrics = len(dr_metrics)
    # print("Aggregate Readiness Scores:")
    # for i in range(num_clients):
    #     aggregate_dr_scores[i] = aggregate_dr_scores[i] / num_metrics
    #     print(f"  Client {i + 1}: {round(aggregate_dr_scores[i], 2)}")
    
    # print("Overall Readiness Score in Federation: ",sum(aggregate_dr_scores)/num_clients)
    # Save the PDF
    pdf.output(output_filename)
    print("------- Data Characteristics Report Generated ----------")