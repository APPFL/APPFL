import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

def find_latest_logs(directory, client_prefix="Client"):
    client_logs = {}
    for file_name in os.listdir(directory):
        if file_name.startswith("result_") and client_prefix in file_name:
            match = re.search(r"_(Client\d+)_(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})\.txt", file_name)
            if match:
                client_id = match.group(1)
                timestamp = datetime.strptime(match.group(2), "%Y-%m-%d-%H:%M:%S")
                file_path = os.path.join(directory, file_name)
                if client_id not in client_logs or timestamp > client_logs[client_id][1]:
                    client_logs[client_id] = (file_path, timestamp)
    return {client_id: info[0] for client_id, info in client_logs.items()}

def extract_log_data(log_file):
    """
    Extract validation accuracy for each round from a client log file.
    Handles cases where `Val Accuracy` is the last number on the next line.
    
    Args:
        log_file (str): Path to the log file.
        
    Returns:
        list: Rounds.
        list: Validation accuracies.
    """
    rounds = []
    val_accuracies = []
    val_loss = []
    log_pattern = r'(\d+)\s+([A-Z])\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(log_pattern, line)
            if match:
    
                val_loss.append(float(match.group(6)))

                rounds.append(int(match.group(1)))
                val_accuracies.append(float(match.group(7)))
    return val_accuracies,val_loss

import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def plot_all_clients(log_directory):
    client_logs = find_latest_logs(log_directory)
    
    # Prepare data structures for validation accuracy and loss
    val_accuracies_data = {}
    val_loss_data = {}
    
    # Extract data for all clients
    for client_id, log_file in client_logs.items():
        val_accuracies, val_loss = extract_log_data(log_file)
        if val_accuracies:
            val_accuracies_data[client_id] = val_accuracies
        if val_loss:
            val_loss_data[client_id] = val_loss
    
    # Create output directory for plots
    output_dir = os.path.join(log_directory, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if there's data to plot
    if not val_accuracies_data and not val_loss_data:
        print("No validation data found for plotting.")
        return
    
    # Create figure with two subplots (stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot validation accuracy (Top subplot)
    if val_accuracies_data:
        for client_id, accuracies in val_accuracies_data.items():
            axes[0].plot(accuracies, label=f"Client {client_id}", marker='o')
        
        axes[0].set_title("Validation Accuracy of All Clients")
        axes[0].set_ylabel("Validation Accuracy")
        axes[0].legend(title="Client ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True)

    # Plot validation loss (Bottom subplot)
    if val_loss_data:
        for client_id, loss in val_loss_data.items():
            axes[1].plot(loss, label=f"Client {client_id}", linestyle='--', marker='x', alpha=0.7)
        
        axes[1].set_title("Validation Loss of All Clients")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Validation Loss")
        axes[1].legend(title="Client ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_clients_val_acc_loss.png"))
    plt.close()  # Close figure to free up memory
