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

    for idx, client_id in enumerate(client_ids):
        plots = readiness_report.get('plots', {}).get(client_id, {})
        html_content += add_client_section(client_id, attribute_keys, plots, readiness_report)

    if 'to_combine' in readiness_report:
        to_combine = readiness_report['to_combine']
        html_content += add_combined_section(to_combine)

    return html_content


def add_combined_section(to_combine):
    html_combined_section = ""
    client_feature_space_dict = {}
    client_ids = list(to_combine.keys())

    for client_id in client_ids:
        feature_space = to_combine.get(client_id, {}).get('feature_space_distribution', {})
        
        if 'pca_components' in feature_space:
            client_feature_space_dict[client_id] = {
                'pca_components': feature_space['pca_components'],
                'explained_variance': feature_space['explained_variance']
            }

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
            
    client_html += '</div>'
    return client_html

def get_class_distribution(restructured_report, num_classes=10):
    class_distribution = restructured_report.get('class_distribution', {})
    complete_class_distribution = {}
    
    for client_id, distribution in class_distribution.items():
        client_class_distribution = {i: 0 for i in range(num_classes)}
        
        for class_id, count in distribution.items():
            client_class_distribution[int(class_id)] = count
        
        complete_class_distribution[client_id] = client_class_distribution
    
    return complete_class_distribution
