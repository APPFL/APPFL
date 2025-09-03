#!/bin/bash
# Script to run MNIST memory profiling experiment comparing original vs optimized versions

set -e

# Create output directory
OUTPUT_DIR="./memory_profiles/mnist_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting MNIST memory profiling experiment..."
echo "Output directory: $OUTPUT_DIR"

# Function to run experiment with a specific version
run_experiment() {
    local VERSION=$1
    local USE_OPTIMIZED=$2
    local VERSION_FLAG=""

    if [ "$USE_OPTIMIZED" = true ]; then
        VERSION_FLAG="--use_optimized_version"
    fi

    echo "============================================"
    echo "Running $VERSION version experiment..."
    echo "============================================"

    # Start server in background
    echo "Starting $VERSION server..."
    python memory_profiling/run_server_memray.py \
        --config ./resources/configs/mnist/server_fedavg.yaml \
        --output-dir "$OUTPUT_DIR" \
        $VERSION_FLAG &
    SERVER_PID=$!

    # Wait for server to start
    sleep 5

    # Start two clients in parallel
    echo "Starting $VERSION client 1..."
    python memory_profiling/run_client_memray.py \
        --config ./resources/configs/mnist/client_1.yaml \
        --output-dir "$OUTPUT_DIR" \
        $VERSION_FLAG &
    CLIENT1_PID=$!

    echo "Starting $VERSION client 2..."
    python memory_profiling/run_client_memray.py \
        --config ./resources/configs/mnist/client_2.yaml \
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

    echo "$VERSION version experiment completed!"
    echo ""
}

# Run original version first
run_experiment "ORIGINAL" false

# Wait a bit between experiments
sleep 10

# Run optimized version
run_experiment "OPTIMIZED" true

echo "============================================"
echo "Both MNIST experiments completed!"
echo "============================================"

# Run analysis automatically
echo "Running memory profile analysis..."
python memory_profiling/analyze_profiles.py "$OUTPUT_DIR"

echo ""
echo "MNIST memory profiling comparison complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  Original version profiles:"
echo "    - server_original_memory_profile.bin"
echo "    - client_1_original_memory_profile.bin"
echo "    - client_2_original_memory_profile.bin"
echo ""
echo "  Optimized version profiles:"
echo "    - server_optimized_memory_profile.bin"
echo "    - client_1_optimized_memory_profile.bin"
echo "    - client_2_optimized_memory_profile.bin"
echo ""
echo "  Analysis files in: $OUTPUT_DIR/analysis/"
echo "    - Flamegraphs (HTML files for browser viewing)"
echo "    - Summary statistics (TXT files)"
echo "    - Allocation tables (TXT files)"
echo ""
echo "Compare the flamegraphs and summaries to see memory optimization improvements!"
