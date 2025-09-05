# APPFL TES Client Docker Image
# This image provides the runtime environment for APPFL clients in TES containers

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy APPFL source code
COPY ../../ /appfl-source/

# Install APPFL from local source (includes most dependencies)
RUN cd /appfl-source && pip install --no-cache-dir -e ".[examples]"

# Install additional TES-specific dependencies
RUN pip install --no-cache-dir requests scipy pyyaml pickle5

# Create necessary directories
RUN mkdir -p /tmp /app/configs /app/data

# Copy APPFL client runner
COPY run_tes_client.py /app/
RUN chmod +x /app/run_tes_client.py

# Set environment variables
ENV PYTHONPATH=/app

# Default command (will be overridden by TES task)
CMD ["python", "/app/run_tes_client.py", "--help"]