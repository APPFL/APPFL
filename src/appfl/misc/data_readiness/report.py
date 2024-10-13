import json
import os

from .plots import generate_combined_feature_space_plot

def get_unique_file_path(output_dir, output_filename, extension):
    file_path = os.path.join(output_dir, f"{output_filename}.{extension}")
    file_counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{output_filename}_{file_counter}.{extension}")
        file_counter += 1
    return file_path

def save_json_report(file_path, readiness_report, logger):
    with open(file_path, 'w') as json_file:
        json.dump(readiness_report, json_file, indent=4)
    logger.info(f"Data readiness report saved as JSON to: {file_path}")

def save_html_report(file_path, html_content, logger):
    with open(file_path, 'w') as html_file:
        html_file.write(html_content)
    logger.info(f"Data readiness report saved as HTML: {file_path}")

def generate_html_content(readiness_report):
    html_content = get_html_header()
    attribute_keys = [key for key in readiness_report.keys() if key != 'to_combine']
    client_ids = list(readiness_report[attribute_keys[0]].keys())

    # Process individual client sections
    for idx, client_id in enumerate(client_ids):
        plots = readiness_report.get('plots', {}).get(client_id, {})
        
        # Add individual client section (assuming _add_client_section handles the plots and other data)
        html_content += add_client_section(client_id, attribute_keys, plots, readiness_report)

    # Process and add the combined PCA section from `to_combine`
    if 'to_combine' in readiness_report:
        to_combine = readiness_report['to_combine']
        html_content += add_combined_section(to_combine)

    return html_content


def add_combined_section(to_combine):
    """Generates the combined PCA section from `to_combine` and returns HTML content."""
    
    html_combined_section = ""
    client_feature_space_dict = {}
    client_ids = list(to_combine.keys())

    # Collect PCA components and explained variance for combined clients
    for client_id in client_ids:
        feature_space = to_combine.get(client_id, {}).get('feature_space_distribution', {})
        
        # Collect PCA components and explained variance from `to_combine`
        if 'pca_components' in feature_space:
            client_feature_space_dict[client_id] = {
                'pca_components': feature_space['pca_components'],
                'explained_variance': feature_space['explained_variance']
            }

    # Generate combined PCA plot and HTML if there is data
    if client_feature_space_dict:
        combined_plot_base64 = generate_combined_feature_space_plot(client_feature_space_dict, client_ids)
        
        html_combined_section = f'''
        <div class="plot">
            <h2>Combined PCA Distribution</h2>
            <img src="data:image/png;base64,{combined_plot_base64}" alt="Combined PCA Distribution">
        </div>
        '''

    return html_combined_section


def get_html_header():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Readiness Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
            h1 { text-align: center; color: #333; }
            .client-section { margin-bottom: 20px; padding: 10px; background-color: #fff; border: 1px solid #ccc; border-radius: 5px; }
            .client-id { font-weight: bold; margin-bottom: 5px; }
            .attribute { margin-left: 10px; color: #555; }
            .plot { text-align: center; margin-top: 20px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Data Readiness Report</h1>
    """

def add_client_section(client_id, attributes, plots, readiness_report):
    client_html = f'<div class="client-section"><div class="client-id">Client ID: {client_id}</div>'
    
    for key in attributes:
        if key == 'plots':
            continue
        value = readiness_report[key][client_id]
        value_str = str(value) if not isinstance(value, list) else ', '.join(map(str, value))
        client_html += f'<div class="attribute"><strong>{key}</strong>: {value_str}</div>'
    
    if plots:
        for plot_name, plot_base64 in plots.items():
            client_html += f'<div class="plot"><h2>{plot_name.replace("_", " ").title()}</h2><img src="data:image/png;base64,{plot_base64}" alt="{plot_name.replace("_", " ").title()}"></div>'
            
    client_html += '</div>'  # Close the client-section div
    return client_html

def get_class_distribution(restructured_report, num_classes=10):
    """
    Extract and complete the class distribution from the report.
    
    Parameters:
    - restructured_report: Dictionary containing class distributions per client.
    - num_classes: Total number of classes (default is 10).
    
    Returns:
    - A dictionary where each client ID has a complete class distribution (missing classes filled with 0), with integer keys for class labels.
    """
    class_distribution = restructured_report.get('class_distribution', {})
    
    # Create a dictionary to store the completed class distribution for each client
    complete_class_distribution = {}
    
    for client_id, distribution in class_distribution.items():
        # Initialize a dictionary with integer keys for all classes, set to 0
        client_class_distribution = {i: 0 for i in range(num_classes)}
        
        # Update the class distribution with the actual values from the report
        for class_id, count in distribution.items():
            client_class_distribution[int(class_id)] = count  # Ensure class_id is treated as an integer
        
        # Add this client's completed distribution to the final dictionary
        complete_class_distribution[client_id] = client_class_distribution
    
    return complete_class_distribution
