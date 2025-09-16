# TES S3 Storage Example

This configuration demonstrates APPFL TES integration with **S3 storage** for remote TES deployments using secure presigned URLs.

## Prerequisites

### 1. AWS S3 Bucket
```bash
aws s3 mb s3://your-appfl-tes-bucket --region us-east-1
```

### 2. AWS Credentials (Server Only)
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

## Configuration

### Server (S3 Storage)
```yaml
server_configs:
  comm_configs:
    tes_configs:
      file_storage: "s3"
      file_storage_kwargs:
        s3_bucket: "your-appfl-tes-bucket"
        s3_region: "us-east-1"
        presigned_url_expiry: 3600  # 1 hour
```

### Clients (Remote TES Endpoints)
```yaml
comm_configs:
  tes_configs:
    tes_endpoint: "https://remote-tes-server.com"
    auth_token: "${TES_AUTH_TOKEN}"
```

## Usage

1. **Update Configuration**: Set your S3 bucket in `server.yaml` and TES endpoints in `clients.yaml`

2. **Set Environment**: Export AWS credentials and TES auth tokens

3. **Run Example**:
```bash
cd examples
python tes/run.py --server_config ./resources/config_tes/simple_net_s3/server.yaml \
                  --client_config ./resources/config_tes/simple_net_s3/clients.yaml
```

## How S3 Transfer Works

- **Server → Client**: Server uploads models to S3, provides presigned download URLs to TES
- **Client → Server**: Server generates presigned upload URLs, clients upload results directly to S3
- **Security**: Clients only get time-limited upload permissions, no AWS credentials needed

## Benefits

- ✅ Works with remote TES endpoints across networks
- ✅ Secure transfer using presigned URLs  
- ✅ No AWS credentials needed in client containers
- ✅ Automatic S3 cleanup after completion