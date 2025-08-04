# APPFL TES Client Docker Image
# This image provides the runtime environment for APPFL clients in TES containers

FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install APPFL
RUN pip install --no-cache-dir appfl[examples]

# Create necessary directories
RUN mkdir -p /tmp /app/configs /app/data

# Copy APPFL client runner
COPY run_tes_client.py /app/
RUN chmod +x /app/run_tes_client.py

# Copy default client configuration
COPY client_tes.yaml /app/configs/

# Set environment variables
ENV PYTHONPATH=/app
ENV APPFL_CONFIG_PATH=/app/configs/client_tes.yaml

# Default command (will be overridden by TES task)
CMD ["python", "/app/run_tes_client.py", "--help"]