from .drmetrics.imbalance_degree import imbalance_degree
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np

def generate_readiness_report(cfg,X, sens_attr, train_datasets, dr_metrics, output_file):
    dr_metrics = ','.join(dr_metrics)

    # Create a PDF canvas
    c = canvas.Canvas(output_file)

    # Set the page margins
    left_margin = 50
    right_margin = 550
    top_margin = 800
    bottom_margin = 50

    # Set the default font and font size
    default_font = "Times-Roman"
    default_font_size = 10

    # Write the report header
    c.setFont("Times-Bold", 16)  # Set the font to bold and increase the font size
    c.drawString(left_margin, top_margin, "Data Readiness Report")

    # Set the font back to normal and decrease the font size
    c.setFont(default_font, default_font_size)

    y_position = top_margin - 50  # Initial y-coordinate for the report sections

    if "ci" in dr_metrics:
        c.drawString(left_margin, y_position, "Class Imbalance Degree:")
        y_position -= 20  # Adjusted y-coordinate for the next line

        c.drawString(left_margin, y_position, "Calculate class proportions and their balance distribution. Evaluate the Euclidean distance; a value closer to 0 suggests balance.")
        y_position -= 20  # Adjusted y-coordinate for the next section

        for i, dataset in enumerate(train_datasets):
            labels = dataset.data_label.tolist()
            imb_deg = imbalance_degree(labels)
            if y_position < bottom_margin + 50:
                c.showPage()  # Move to a new page if content goes beyond the bottom margin
                y_position = top_margin
                c.setFont(default_font, default_font_size)  # Set the font back to default on the new page
            c.drawString(left_margin + 20, y_position - i * 20, f"  Client {i + 1}: {round(imb_deg, 2)}")
            if imb_deg > 0.27:
                c.drawString(left_margin + 100, y_position - i * 20, "*(Imbalance Detected, this client will be removed.)")
                train_datasets.remove(dataset)
                cfg.num_clients -= 1
                
        y_position -= (len(train_datasets) + 1) * 20  # Adjusted y-coordinate for the next section

    if "ss" in dr_metrics:
        c.drawString(left_margin, y_position, "Sample Size:")
        y_position -= 20  # Adjusted y-coordinate for the next line
        c.drawString(left_margin, y_position, "Calculate the number of samples in each client's dataset.")
        y_position -= 20  # Adjusted y-coordinate for the next line
        for i, dataset in enumerate(train_datasets):
            sample_size = len(dataset.data_label.tolist())
            if y_position < bottom_margin + 50:
                c.showPage()  # Move to a new page if content goes beyond the bottom margin
                y_position = top_margin
                c.setFont(default_font, default_font_size)  # Set the font back to default on the new page
            c.drawString(left_margin + 20, y_position - i * 20, f"  Client {i + 1}: {round(sample_size)}")
        y_position -= (len(train_datasets) + 1) * 20  # Adjusted y-coordinate for the next section

    if "rr" in dr_metrics:
        num_clients = len(train_datasets)
        c.drawString(left_margin, y_position, "Representation Rate:")
        y_position -= 20  # Adjusted y-coordinate for the next line
        c.drawString(left_margin, y_position, "Calculate the representation rate of sensitive attributes in each client's dataset.")

        # Divide the dataset into num_clients partitions randomly
        np.random.seed(42)  # Set seed for reproducibility
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        partition_size = len(X) // len(train_datasets)
        partitions = [indices[i:i + partition_size] for i in range(0, len(X), partition_size)]

        # Calculate the representation rate for each sensitive attribute
        # Iterate over each sensitive attribute
        for attr in sens_attr:
            y_position -= 20
            c.drawString(left_margin, y_position, f"  Sensitive Attribute: {attr}")

            # Iterate over each partition
            for i, partition in enumerate(partitions):
                if y_position < bottom_margin + 50:
                    c.showPage()  # Move to a new page if content goes beyond the bottom margin
                    y_position = top_margin
                    c.setFont(default_font, default_font_size)  # Set the font back to default on the new page
                y_position -= 15
                c.drawString(left_margin + 20, y_position, f"    Client {i + 1}:")

                # Count occurrences of each unique value in the current partition for the sensitive attribute
                unique_values, counts = np.unique(X[attr][partition], return_counts=True)

                # Calculate and display representation rate for each unique value in the current partition
                for value, count in zip(unique_values, counts):
                    representation_rate = count / len(partition)
                    y_position -= 15
                    c.drawString(left_margin + 40, y_position, f"{value}: {representation_rate:.2%}")
                    if y_position < bottom_margin:
                        c.showPage()  # Move to a new page if content goes beyond the bottom margin
                        y_position = top_margin
                        c.setFont(default_font, default_font_size)  # Set the font back to default on the new page

        # Adjust y-coordinate for any additional content
        y_position -= 20

    if "ds" in dr_metrics:
        c.drawString(left_margin, y_position, "Data Sparsity:")
        y_position -= 20
        c.drawString(left_margin, y_position, "Calculate the data sparsity in each client's dataset.")
        y_position -= 20
        for i, dataset in enumerate(train_datasets):
            sparsity = np.mean(np.isnan(dataset.data_input.numpy()))
            if y_position < bottom_margin + 50:
                c.showPage()
                y_position = top_margin
                c.setFont(default_font, default_font_size)
            c.drawString(left_margin + 20, y_position - i * 20, f"  Client {i + 1}: {round(sparsity, 2)}")
        y_position -= (len(train_datasets) + 1) * 20

    if "dup" in dr_metrics:
        c.drawString(left_margin, y_position, "Duplicate Records:")
        y_position -= 20
        c.drawString(left_margin, y_position, "Calculate the number of duplicate records in each client's dataset.")
        y_position -= 20
        for i, dataset in enumerate(train_datasets):
            duplicates = len(dataset.data_input) - len(np.unique(dataset.data_input.numpy(), axis=0))
            if y_position < bottom_margin + 50:
                c.showPage()
                y_position = top_margin
                c.setFont(default_font, default_font_size)
            c.drawString(left_margin + 20, y_position - i * 20, f"  Client {i + 1}: {round(duplicates)}")
        y_position -= (len(train_datasets) + 1) * 20  

    if "dc" in dr_metrics:
        c.drawString(left_margin, y_position, "Data Consistency:")
        y_position -= 20
        c.drawString(left_margin, y_position, "Consistency is measured as the mean of the standard deviation of the correlation matrix.")
        y_position -= 20
        for i, dataset in enumerate(train_datasets):
            input_matrix = dataset.data_input.numpy()
            flattened_matrix = input_matrix.reshape(input_matrix.shape[0], -1)
            correlation_matrix = np.corrcoef(flattened_matrix, rowvar=False)
            consistency = np.mean(np.std(correlation_matrix, axis=0))
            if y_position < bottom_margin + 50:
                c.showPage()
                y_position = top_margin
                c.setFont(default_font, default_font_size)
            c.drawString(left_margin + 20, y_position - i * 20, f"  Client {i + 1}: {round(consistency, 2)}")
        y_position -= (len(train_datasets) + 1) * 20

    if "dd" in dr_metrics:
        c.drawString(left_margin, y_position, "Data Distribution:")
        y_position -= 20
        c.drawString(left_margin, y_position, "Distribution is measured as the mean of the standard deviation of the input data.")
        y_position -= 20
        for i, dataset in enumerate(train_datasets):
            input_matrix = dataset.data_input.numpy()
            flattened_matrix = input_matrix.reshape(input_matrix.shape[0], -1)
            distribution = np.mean(np.std(flattened_matrix, axis=0))
            distribution = round(float(distribution), 2)
            
            if y_position < bottom_margin + 50:
                c.showPage()
                y_position = top_margin
                c.setFont(default_font, default_font_size)

            c.drawString(left_margin + 20, y_position - i * 20, f"  Client {i + 1}: {distribution}")
        y_position -= (len(train_datasets) + 1) * 20

    # Adjust y-coordinate for any additional content
    y_position -= 20

    # Save the PDF file
    c.save()
