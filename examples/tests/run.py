import subprocess
import yaml
import itertools
import os
import shutil
import glob
import logging

# Set up logging
log_file = "experiment_log.txt"
logger = logging.getLogger()

# Set log level
logger.setLevel(logging.DEBUG)

# Clear existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# Add a file handler to log to a file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Add a stream handler to log to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Only INFO level and higher for console output
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(stream_handler)

logger.info("Experiment started")

# Define the range of values to test
num_clients_list = [2, 4, 8, 10]
noise_props = [0,0.25, 0.5, 0.75, 0.95]
sample_sizes = [100, 500, 1000, 3000, 6000]

def get_dataset_name(client_config_path):
    """Extract dataset name from client config YAML."""
    logger.info(f"Reading dataset name from config: {client_config_path}")
    try:
        with open(client_config_path, 'r') as file:
            config = yaml.safe_load(file)
        dataset_name = config.get("data_configs", {}).get("dataset_name", "unknown_dataset")
        logger.info(f"Dataset name extracted: {dataset_name}")
        return dataset_name
    except Exception as e:
        logger.error(f"Error reading dataset name from {client_config_path}: {e}")
        raise

def get_algorithm_name(server_config_path):
    """Extract algorithm name from server config YAML."""
    logger.info(f"Reading algorithm name from config: {server_config_path}")
    try:
        with open(server_config_path, 'r') as file:
            config = yaml.safe_load(file)
        algorithm_name = config.get("server_configs", {}).get("aggregator", "unknown_algorithm")
        logger.info(f"Algorithm name extracted: {algorithm_name}")
        return algorithm_name
    except Exception as e:
        logger.error(f"Error reading algorithm name from {server_config_path}: {e}")
        raise

def modify_yaml(file_path, key_path, value):
    """Modify a specific value in a nested YAML file without altering structure."""
    logger.info(f"Modifying YAML file: {file_path}, key_path: {key_path}, new value: {value}")
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        keys = key_path.split('.')
        sub_config = config
        for key in keys[:-1]:
            if key in sub_config:
                sub_config = sub_config[key]
            else:
                raise KeyError(f"Key path '{key_path}' not found in YAML file.")
        
        sub_config[keys[-1]] = value

        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        logger.info(f"YAML file modified successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error modifying YAML file {file_path}: {e}")
        raise

def run_experiment(num_clients, noise_props, sample_sizes, server_config, client_config):
    """Run the MPI experiment with given parameters and save output in structured folders."""
    dataset_name = get_dataset_name(client_config)
    algorithm_name = get_algorithm_name(server_config)
    output_dir = f"tests/{algorithm_name}/{dataset_name}/num_clients_{num_clients}/noise_props_{noise_props}/sample_size_{sample_sizes}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "output.log")
    
    command = [
        "mpiexec", "-n", str(num_clients+1), "python", "mpi/run_mpi.py",
        "--server_config", server_config,
        "--client_config", client_config
    ]
    logger.info(f"Running experiment with num_clients={num_clients}, noise_props={noise_props}, sample_size={sample_sizes}")
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        with open(output_file, "w") as f:
            subprocess.run(command, stdout=f, stderr=f)
        logger.info(f"Experiment output logged to: {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running experiment command: {e}")
        raise

    # After the experiment, check for the generated HTML files in the 'outputs/' directory
    html_files = glob.glob("output/*.html")  # Adjust this if needed
    if html_files:
        # Sort the HTML files by modification time (most recent first)
        latest_html_file = max(html_files, key=os.path.getmtime)
        logger.info(f"Latest HTML file found: {latest_html_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the latest HTML file to the results folder
        try:
            shutil.copy(latest_html_file, output_dir)
            logger.info(f"Latest HTML file copied to: {output_dir}")
        except Exception as e:
            logger.error(f"Error copying HTML file {latest_html_file} to {output_dir}: {e}")
            raise
    else:
        logger.warning("No HTML files found in 'outputs/' directory.")
    
    png_files = glob.glob("output/plots/*.png")  # Adjust this if needed
    if png_files:
        # Sort the PNG files by modification time (most recent first)
        latest_png_file = max(png_files, key=os.path.getmtime)
        logger.info(f"Latest PNG file found: {latest_png_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the latest PNG file to the results folder
        try:
            shutil.copy(latest_png_file, output_dir)
            logger.info(f"Latest PNG file copied to: {output_dir}")
        except Exception as e:
            logger.error(f"Error copying PNG file {latest_png_file} to {output_dir}: {e}")
            raise

# Iterate over all combinations of parameters
server_config_path = "resources/configs/mnist/server_fedyogi.yaml"
client_config_path = "resources/configs/mnist/client_1.yaml"

for num_clients, noise_props, sample_sizes in itertools.product(num_clients_list, noise_props, sample_sizes):
    logger.info(f"Starting experiment with num_clients={num_clients}, noise_props={noise_props}, sample_size={sample_sizes}")
    try:
        modify_yaml(client_config_path, "data_configs.dataset_kwargs.num_clients", num_clients)
        modify_yaml(client_config_path, "data_configs.dataset_kwargs.sample_size", sample_sizes)
        modify_yaml(client_config_path, "data_configs.dataset_kwargs.noise_prop", noise_props)
        modify_yaml(server_config_path, "server_configs.num_clients", num_clients)
        run_experiment(num_clients, noise_props, sample_sizes, server_config_path, client_config_path)
        logger.info(f"Experiment with num_clients={num_clients}, noise_props={noise_props}, sample_size={sample_sizes} completed successfully.")
    except Exception as e:
        logger.error(f"Error during experiment with num_clients={num_clients}, noise_props={noise_props}, sample_size={sample_sizes}: {e}")

logger.info("Experiment completed")
