#!/bin/bash

# APPFL TES Testing Setup Script
# This script sets up a local testing environment for GA4GH TES integration

echo "Setting up APPFL TES testing environment..."

# 1. Install Funnel (local TES server)
echo "Installing Funnel TES server..."
curl -L https://github.com/ohsu-comp-bio/funnel/releases/download/0.10.1/funnel-linux-amd64-0.10.1.tar.gz | tar xz
sudo mv funnel /usr/local/bin/

# 2. Create Funnel configuration
echo "Creating Funnel configuration..."
mkdir -p ~/.funnel
cat > ~/.funnel/funnel.config.yml << EOF
Server:
  HTTPPort: 8000
  RPCPort: 9090

Compute: local

LocalStorage:
  AllowedDirs:
    - /tmp
    - $(pwd)

Database: boltdb
EOF

# 3. Build APPFL TES client Docker image
echo "Building APPFL TES client Docker image..."
cd $(dirname "$0")
docker build -t appfl/tes-client:latest .

# 4. Start Funnel server in background
echo "Starting Funnel TES server..."
funnel server run --config ~/.funnel/funnel.config.yml &
FUNNEL_PID=$!
echo "Funnel PID: $FUNNEL_PID"

# Wait for server to start
sleep 5

# 5. Test TES server connectivity
echo "Testing TES server connectivity..."
curl -s http://localhost:8000/ga4gh/tes/v1/service-info | jq .

if [ $? -eq 0 ]; then
    echo "✅ Funnel TES server is running successfully!"
    echo "TES endpoint: http://localhost:8000"
    echo "To stop the server: kill $FUNNEL_PID"
else
    echo "❌ Failed to start Funnel TES server"
    kill $FUNNEL_PID 2>/dev/null
    exit 1
fi

# 6. Create test environment variables
cat > test_env.sh << EOF
#!/bin/bash
export TES_ENDPOINT="http://localhost:8000"
export TES_AUTH_TOKEN=""  # No auth needed for local testing
export DOCKER_IMAGE="appfl/tes-client:latest"
echo "Test environment configured!"
echo "Run: source test_env.sh"
EOF

chmod +x test_env.sh

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Run: source test_env.sh"
echo "2. Run: python test_tes_basic.py"
echo "3. Run: python test_tes_federated.py"