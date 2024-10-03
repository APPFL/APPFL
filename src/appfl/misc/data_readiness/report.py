import json
import os


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
    attribute_keys = list(readiness_report.keys())
    client_ids = list(readiness_report[attribute_keys[0]].keys())

    for client_id in client_ids:
        plots = readiness_report.get('plots', {}).get(client_id, {})
        html_content += add_client_section(client_id, attribute_keys, plots, readiness_report)

    html_content += "</body></html>"
    return html_content

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