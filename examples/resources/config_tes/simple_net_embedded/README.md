# TES Data Embedding Demo

This approach embeds client datasets directly into Docker images during build time, avoiding the need for runtime volume mounting which isn't supported by standard TES/Funnel.

## Approach

Instead of mounting volumes at runtime, we:
1. **Generate datasets** on the host filesystem  
2. **Build client-specific images** with data baked in
3. **Use different images** for each client in TES tasks

## Setup

### 1. Generate Test Datasets
```bash
cd examples/resources/config_tes/simple_net_embedded
python generate_test_data.py
```

### 2. Build Client-Specific Images
```bash
# Make script executable
chmod +x build_client_images.sh

# Build images with embedded data
./build_client_images.sh
```

This creates:
- `appfl/client1:data-embedded` (contains client_0 data)
- `appfl/client2:data-embedded` (contains client_1 data)

### 3. Start Funnel
```bash
mkdir -p /tmp/funnel-workspace
cd /tmp/funnel-workspace
funnel server run
```

## How It Works

### Build Process:
```dockerfile
# Dockerfile.client1
FROM appfl/client:latest
COPY /tmp/tes-data/client_0/ /data/    # Data baked into image
ENV DATA_DIR=/data
```

### Runtime:
- **TESClient1** uses `appfl/client1:data-embedded` 
- **TESClient2** uses `appfl/client2:data-embedded`
- Each container has `/data` with client-specific datasets

### Data Flow:
```
Host: /tmp/tes-data/client_0/  
  ↓ (COPY during build)
Image: appfl/client1:data-embedded with /data
  ↓ (TES container starts)  
Container: /data/client_0_features.csv available
  ↓ (file_dataset.py loads)
FL Client: Different datasets per client
```

## Testing

### Run with Data-Embedded Images
```bash
cd ../../../tes
python run.py --client_config ./resources/config_tes/simple_net_embedded/clients.yaml
```

### Verify Data Loading
```bash
# Check what data is in each image
docker run --rm appfl/client1:data-embedded ls -la /data
docker run --rm appfl/client2:data-embedded ls -la /data

# Check file contents
docker run --rm appfl/client1:data-embedded head /data/metadata.json
docker run --rm appfl/client2:data-embedded head /data/metadata.json
```

## Expected Results

- **TESClient1**: Loads 100 samples from embedded client_0 data
- **TESClient2**: Loads 150 samples from embedded client_1 data  
- **Different sample sizes**: Server should receive different `sample_size` values
- **No volume mounting errors**: Images contain data, no runtime mounting needed

## Advantages

✅ **Works with standard TES/Funnel** - no volume mounting needed  
✅ **Data isolation** - each client has only their data  
✅ **Portable** - images can run anywhere with data included  
✅ **Secure** - data is read-only and isolated per client  

## Disadvantages  

❌ **Larger images** - data increases image size  
❌ **Build-time dependency** - need data available during build  
❌ **Less flexible** - changing data requires rebuilding images  
❌ **Data duplication** - data exists in both host and image  

## Production Considerations

For production federated learning:
- **Sensitive data**: Ensure proper image security and access controls
- **Large datasets**: Consider hybrid approaches (small data embedded, large data via network)
- **Dynamic data**: Use this for static datasets, alternative approaches for changing data
- **Image management**: Implement proper versioning and cleanup of client images