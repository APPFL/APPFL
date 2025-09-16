# TES MNIST S3 Example

APPFL TES integration with **MNIST dataset and S3 storage** for remote federated learning.

## Prerequisites

```bash
# S3 bucket
aws s3 mb s3://your-appfl-tes-bucket --region us-east-1

# AWS credentials (server only)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Docker image
docker build -f examples/tes/Dockerfile -t appfl/client:latest .
```

## Configuration

- **Model**: CNN for MNIST (28x28 images, 10 classes)
- **Data**: Non-IID class partitioning between 2 clients
- **Storage**: S3 with presigned URLs for secure transfer

## Usage

1. **Configure**: Update S3 bucket in `server.yaml`
2. **Run**:
```bash
cd examples
python tes/run.py --server_config ./resources/config_tes/mnist_s3/server.yaml \
                  --client_config ./resources/config_tes/mnist_s3/clients.yaml
```

## Benefits

- ✅ Realistic FL with MNIST + CNN
- ✅ Remote TES deployment via S3
- ✅ Non-IID data distribution
- ✅ Secure presigned URL transfers