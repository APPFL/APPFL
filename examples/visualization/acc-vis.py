"""
Visualizing several experiment runs of different algorithms
"""
import os
import pickle
import argparse
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

def plot_mean_var(datasets, color, label):
    """Plot mean and var for one set of data"""
    # Define the time points at which we want to interpolate, we are taking the maximum end time here
    time_points = np.arange(0, np.max([np.max(data[0]) for data in datasets]))

    # Initialize list to hold interpolated accuracy values for all datasets
    interpolated_accuracies = []

    # Interpolate each dataset and evaluate at the desired time points
    for data in datasets:
        time = data[0]
        accuracy = data[1]
        interp_func = interpolate.interp1d(time, accuracy, kind='linear', fill_value="extrapolate")
        interpolated_accuracies.append(interp_func(time_points))

    # Convert to a 2D numpy array for easier manipulation
    interpolated_accuracies = np.array(interpolated_accuracies)

    # Calculate the mean and standard deviation of the accuracy at each time point
    mean_accuracies = np.mean(interpolated_accuracies, axis=0)
    std_accuracies = np.std(interpolated_accuracies, axis=0)

    # Plot mean accuracy with standard deviation as shaded region
    plt.plot(time_points, mean_accuracies, label=f'{label} Mean', color=color)
    plt.fill_between(time_points, mean_accuracies - std_accuracies, mean_accuracies + std_accuracies, color=color, alpha=0.2, label=f'{label} Standard deviation')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Mean and Standard Deviation of Accuracy over Time')

def plot_mean_var_among_datasets(datasets_list, colors, labels, save_path=None):
    """Plot mean and var for different sets of data"""
    plt.figure(figsize=(10,6))
    for i in range(len(datasets_list)):
        plot_mean_var(datasets_list[i], colors[i], labels[i])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_dirs(dirs, colors, save_path=None, labels=None):
    # Initialize an empty list to hold the data
    data = []

    # Walk through all files in the directory, including subdirectories
    for directory in dirs:
        # Initialize an empty list to hold the data for this directory path
        dirpath_data = []
            
        # Walk through all files in the directory
        for dirpath, dirnames, filenames in os.walk(directory):
            # For each file in the current directory
            for filename in filenames:
                # If the file starts with 'metric' and ends with '.pkl'
                if filename.startswith('metric') and filename.endswith('.pkl'):
                    # Construct the full file path
                    filepath = os.path.join(dirpath, filename)
                    # Load the file and append the data to the list
                    with open(filepath, 'rb') as f:
                        dirpath_data.append(pickle.load(f))

            # Append the list of data for this directory path to the main dictionary
            if dirpath_data != []: 
                data.append(dirpath_data)

    plot_mean_var_among_datasets(data, colors, labels, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for the visualization result')
    parser.add_argument('--output_filename', type=str, required=True, help='file name of the output visualization file')
    parser.add_argument('--raw_result_dirs', type=str, nargs='+', required=True, help='directories containing raw experiment results')
    parser.add_argument('--labels', type=str, nargs='+', required=True, help='labels for different experiments')
    args = parser.parse_args()
    assert len(args.raw_result_dirs) == len(args.labels), "Number of labels and number of experiment directories should be the same"
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'lightblue', 'lightgreen', 'lightyellow', 'gray', 'purple', 'pink', 'orange', 'darkgreen', 'navy', 'salmon', 'gold', 'darkred']
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.output_filename}.pdf')
    plot_dirs(args.raw_result_dirs, colors, save_path, args.labels)