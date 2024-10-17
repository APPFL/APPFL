# Base image
FROM python:3.10

# Install system dependencies (Open MPI)
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    && apt-get clean

# Set the working directory
WORKDIR /app

# COPY all docs from  ../../.. to /app
COPY . /app

# Install the dependencies
RUN pip install -e ".[examples]"
RUN pip install globus-compute-sdk==2.26.0
RUN pip install globus-compute-endpoint==2.26.0

# Change directory to examples
WORKDIR /app/examples

# Run the app
ENTRYPOINT ["python", "globus_compute/run.py"]
