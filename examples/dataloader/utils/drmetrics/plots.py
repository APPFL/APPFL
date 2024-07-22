import torch
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Function to plot subplots of class distributions and save to PDF
def plot_class_distribution_subplots_to_pdf(labels_list, pdf, num_clients):
    cols = 2  # Number of columns for subplots
    rows = (num_clients + cols - 1) // cols  # Compute the number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    class_labels = set()
    
    for labels in labels_list:
        class_labels.update(labels)
    class_labels = sorted(list(class_labels))

    for i, labels in enumerate(labels_list):
        counts = torch.bincount(torch.tensor(labels), minlength=len(class_labels))
        
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        wedges, _ = ax.pie(counts, startangle=140)
        ax.set_title(f'Client {i + 1} Class Distribution')
        ax.axis('equal')

    # Remove empty subplots
    if num_clients % cols != 0:
        for j in range(num_clients, rows * cols):
            fig.delaxes(axes[j // cols, j % cols] if rows > 1 else axes[j % cols])

    # Add legend
    fig.legend(wedges, class_labels, title="Classes", loc="center right")
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save plot to a temporary file
    temp_file = f"temp_class_distribution_subplots_with_legend.png"
    plt.savefig(temp_file)
    plt.close()

    # Add plot to PDF
    pdf.image(temp_file, x=10, y=None, w=180)
    pdf.ln()
    
    # Delete temporary file
    import os
    os.remove(temp_file)

# Function to plot KDE and save to PDF
def plot_kde_to_pdf(data_list, pdf, num_clients):
    cols = 2  # Number of columns for subplots
    rows = (num_clients + cols - 1) // cols  # Compute the number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    
    for i, data in enumerate(data_list):
        flattened_data = data.flatten().numpy()
        mean = torch.mean(data).item()
        std = torch.std(data).item()
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        sns.kdeplot(flattened_data, bw_adjust=0.5, ax=ax)
        ax.set_title(f'Client {i + 1} Data Distribution \nMean: {mean:.2f}, Std: {std:.2f}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Density')

    # Remove empty subplots
    if num_clients % cols != 0:
        for j in range(num_clients, rows * cols):
            fig.delaxes(axes[j // cols, j % cols] if rows > 1 else axes[j % cols])

    plt.tight_layout()

    # Save plot to a temporary file
    temp_file = f"temp_kde_subplots.png"
    plt.savefig(temp_file)
    plt.close()

    # Add plot to PDF
    pdf.image(temp_file, x=10, y=None, w=180)
    pdf.ln()

    # Delete temporary file
    import os
    os.remove(temp_file)