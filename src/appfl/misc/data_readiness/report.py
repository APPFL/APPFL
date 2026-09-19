import json
import os
from collections.abc import Mapping
from html import escape
from .calibration import aggregate_calibration_reports
from .plots import generate_combined_feature_space_plot


def get_unique_file_path(output_dir, output_filename, extension):
    file_path = os.path.join(output_dir, f"{output_filename}.{extension}")
    file_counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(
            output_dir, f"{output_filename}_{file_counter}.{extension}"
        )
        file_counter += 1
    return file_path


def save_json_report(file_path, readiness_report, logger):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(readiness_report, json_file, indent=4)
    logger.info(f"Data readiness report saved as JSON to: {file_path}")


def save_html_report(file_path, html_content, logger):
    with open(file_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
    logger.info(f"Data readiness report saved as HTML: {file_path}")


def get_calibration_summary(readiness_report, expected_client_ids=None):
    """Aggregate calibration entries using their transport client IDs."""
    entries = readiness_report.get("specified_metrics", {})
    if not isinstance(entries, Mapping):
        raise ValueError("specified_metrics must map client IDs to metrics")
    if not any(
        isinstance(entry, Mapping) and "subgroup_calibration" in entry
        for entry in entries.values()
    ):
        return None
    client_ids = set(expected_client_ids or ())
    for values in readiness_report.values():
        if isinstance(values, Mapping):
            client_ids.update(values)
    if set(entries) != client_ids or any(
        not isinstance(entry, Mapping) or "subgroup_calibration" not in entry
        for entry in entries.values()
    ):
        raise ValueError("all clients must supply a subgroup calibration report")
    return aggregate_calibration_reports(
        {client: entry["subgroup_calibration"] for client, entry in entries.items()}
    )


def add_calibration_section(summary):
    """Render metrics for retained patients without exposing extra totals."""
    content = "<h2>Subgroup calibration</h2>"
    content += (
        f"<p>{summary['n_bins']} probability bins; minimum cell count "
        f"{summary['min_cell_count']}; minimum events and non-events "
        f"{summary['min_outcome_count']}. Scores describe retained patients only. "
        "Missing groups are unavailable. Full-cohort coverage is unknown.</p>"
        "<p>Wider bins can hide calibration differences. Suppression does not "
        "provide a formal privacy guarantee.</p>"
    )
    if not summary["groups"]:
        return content + "<p>Unavailable: no cells meet the reporting thresholds.</p>"
    content += (
        "<table><thead><tr><th>Group</th><th>Retained patients</th>"
        "<th>ECE</th><th>Brier score</th><th>Warning</th></tr></thead><tbody>"
    )
    for group, metrics in summary["groups"].items():
        content += (
            f"<tr><td>{escape(group)}</td><td>{metrics['n']}</td>"
            f"<td>{metrics['ece']:.6f}</td><td>{metrics['brier']:.6f}</td>"
            f"<td>{escape(metrics['warning'] or '')}</td></tr>"
        )
    return content + "</tbody></table>"


def generate_html_content(readiness_report, calibration_summary=None):
    if calibration_summary is None:
        calibration_summary = get_calibration_summary(readiness_report)
    html_content = get_html_header()

    attribute_keys = [key for key in readiness_report.keys() if key != "to_combine"]
    client_ids = list(readiness_report[attribute_keys[0]].keys())

    # Start creating the table
    html_content += "<table><thead><tr>"

    # Add table headers for each metric
    for key in attribute_keys:
        if key == "plots":
            continue
        if key == "specified_metrics":
            html_content += f'<th style="text-align: center; color: red;" colspan="2">{key.replace("_", " ").title()}</th>'
        else:
            html_content += f"<th>{key.replace('_', ' ').title()}</th>"

    html_content += "</tr></thead><tbody>"

    # Add data for each client
    for client_id in client_ids:
        html_content += "<tr>"
        for key in attribute_keys:
            if key == "plots":
                continue
            value = readiness_report[key][client_id]
            if (
                key == "specified_metrics"
                and isinstance(value, Mapping)
                and "subgroup_calibration" in value
            ):
                value = dict(
                    value, subgroup_calibration="See subgroup calibration below"
                )
            value_str = (
                str(value)
                if not isinstance(value, list)
                else ", ".join(map(str, value))
            )
            html_content += f"<td>{escape(value_str)}</td>"
        html_content += "</tr>"

    html_content += "</tbody></table>"
    if calibration_summary is not None:
        html_content += add_calibration_section(calibration_summary)

    # Add plots for each client
    for client_id in client_ids:
        plots = readiness_report.get("plots", {}).get(client_id, {})
        if plots:
            html_content += add_client_plots(client_id, plots)

    # Add the combined section at the bottom
    if "to_combine" in readiness_report:
        to_combine = readiness_report["to_combine"]
        html_content += add_combined_section(to_combine)

    return html_content


def add_client_plots(client_id, plots):
    """Adds per-client plots below the table and displays them side by side."""
    client_plot_html = (
        f'<div class="client-plots"><h3>Plots for Client ID: {client_id}</h3>'
    )
    client_plot_html += (
        '<div class="plots-row">'  # Flexbox container for side-by-side plots
    )

    for plot_name, plot_base64 in plots.items():
        client_plot_html += f"""
            <div class="plot">
                <h3>{plot_name.replace("_", " ").title()}</h3>
                <img src="data:image/png;base64,{plot_base64}" alt="{plot_name.replace("_", " ").title()}">
            </div>
        """

    client_plot_html += "</div></div>"  # Close the flexbox container and client section
    return client_plot_html


def add_combined_section(to_combine):
    html_combined_section = ""
    client_feature_space_dict = {}
    client_ids = list(to_combine.keys())

    for client_id in client_ids:
        feature_space = to_combine.get(client_id, {}).get(
            "feature_space_distribution", {}
        )
        if "pca_components" in feature_space:
            client_feature_space_dict[client_id] = {
                "pca_components": feature_space["pca_components"],
                "explained_variance": feature_space["explained_variance"],
            }

    if client_feature_space_dict:
        combined_plot_base64 = generate_combined_feature_space_plot(
            client_feature_space_dict, client_ids
        )
        html_combined_section = f"""
            <h3>Combined PCA Distribution</h3>
            <div class="combined-pca-plot">

                <img src="data:image/png;base64,{combined_plot_base64}" alt="Combined PCA Distribution">
            </div>
        """
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
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f4f4f4;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            table, th, td {
                border: 1px solid #ccc;
            }
            th, td {
                padding: 10px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .plot {
                text-align: center;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            img {
                max-width: 100%;
                max-height: 100%;
                height: auto;
            }
            .client-plots {
                margin-bottom: 30px;
            }
            .client-plots h2 {
                text-align: center;
                color: #444;
            }
            .plots-row {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px; /* Spacing between plots */
            }
            .plot {
                width: 300px;
                text-align: center;
            }
            /* Center the combined PCA plot and increase its size */
            .combined-pca-plot {
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 40px 0; /* Add vertical spacing */
            }
            h3{
                text-align: center;
                color: #444;
            }
            .combined-pca-plot img {
                width: 50%; /* Increase the size to 60% of the container */
                height: auto;
                max-width: 1000px; /* Ensure it doesn't exceed a reasonable size */
            }
        </style>
    </head>
    <body>
        <h1>Data Readiness Report</h1>
    """


def get_class_distribution(restructured_report, num_classes=10):
    class_distribution = restructured_report.get("class_distribution", {})
    complete_class_distribution = {}

    for client_id, distribution in class_distribution.items():
        client_class_distribution = {i: 0 for i in range(num_classes)}
        for class_id, count in distribution.items():
            client_class_distribution[int(class_id)] = count
        complete_class_distribution[client_id] = client_class_distribution

    return complete_class_distribution
