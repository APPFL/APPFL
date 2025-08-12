#!/bin/bash
# Script to run ResNet memory profiling experiment with dummy CIFAR-10 data
# This focuses on training memory usage, not data loading

set -e

# Create output directory
OUTPUT_DIR="./memory_profiles/resnet_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting ResNet memory profiling experiment with dummy CIFAR-10..."
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This experiment uses:"
echo "  - ResNet18 model (~11M parameters)"
echo "  - Dummy CIFAR-10 data (64 samples per client)"
echo "  - Focus on training memory, not data loading"
echo ""

# Function to run experiment with a specific version
run_resnet_experiment() {
    local VERSION=$1
    local USE_OPTIMIZED=$2
    local VERSION_FLAG=""
    
    if [ "$USE_OPTIMIZED" = true ]; then
        VERSION_FLAG="--use_optimized_version"
    fi
    
    echo "============================================"
    echo "Running $VERSION version ResNet experiment..."
    echo "============================================"
    
    # Start server in background
    echo "Starting $VERSION server..."
    python memory_profiling/run_server_memray.py \
        --config ./memory_profiling/configs/server_resnet_dummy.yaml \
        --output-dir "$OUTPUT_DIR" \
        $VERSION_FLAG &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Start two clients in parallel
    echo "Starting $VERSION client 1..."
    python memory_profiling/run_client_memray.py \
        --config ./memory_profiling/configs/client_1_resnet_dummy.yaml \
        --output-dir "$OUTPUT_DIR" \
        $VERSION_FLAG &
    CLIENT1_PID=$!
    
    echo "Starting $VERSION client 2..."
    python memory_profiling/run_client_memray.py \
        --config ./memory_profiling/configs/client_2_resnet_dummy.yaml \
        --output-dir "$OUTPUT_DIR" \
        $VERSION_FLAG &
    CLIENT2_PID=$!
    
    # Wait for clients to complete
    echo "Waiting for $VERSION clients to complete..."
    wait $CLIENT1_PID
    wait $CLIENT2_PID
    
    # Stop server
    echo "Stopping $VERSION server..."
    kill $SERVER_PID 2>/dev/null || true
    
    echo "$VERSION version ResNet experiment completed!"
    echo ""
}

# Run original version first
run_resnet_experiment "ORIGINAL" false

# Wait a bit between experiments
sleep 10

# Run optimized version
run_resnet_experiment "OPTIMIZED" true

echo "============================================"
echo "Both ResNet experiments completed!"
echo "============================================"

# Run analysis automatically
echo "Running memory profile analysis..."
python memory_profiling/analyze_profiles.py "$OUTPUT_DIR"

echo ""
echo "ResNet memory profiling comparison complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  Original version profiles:"
echo "    - server_original_memory_profile.bin"
echo "    - client_Client1_original_memory_profile.bin"
echo "    - client_Client2_original_memory_profile.bin"
echo ""
echo "  Optimized version profiles:"
echo "    - server_optimized_memory_profile.bin"
echo "    - client_Client1_optimized_memory_profile.bin"  
echo "    - client_Client2_optimized_memory_profile.bin"
echo ""
echo "Analysis files in: $OUTPUT_DIR/analysis/"
echo ""
echo "Expected optimizations to observe:"
echo "  - Reduced deepcopy operations in model state management"
echo "  - More efficient tensor operations instead of dictionary copying"
echo "  - Better memory cleanup during training iterations"
echo "  - Reduced peak memory usage during parameter updates"
echo ""
echo "Open the flamegraph HTML files to see detailed memory allocation patterns!"