#!/bin/bash

# Create a directory for all installation
cd .. && mkdir -p .compressor && cd .compressor

# Install ZFP
if pip show "zfpy" >/dev/null 2>&1; then
    echo "zfpy is already installed."
else
    # If the package is not installed, install it
    echo "Installing zfpy..."
    pip install zfpy
fi

# Install SZ2
echo "Installing SZ2..."
git clone https://github.com/szcompressor/SZ.git && cd SZ
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=.. ..
make
make install
echo "SZ2 installation done."
echo "======================"
cd ../..

# Install SZ3
echo "Installing SZ3..."
git clone https://github.com/szcompressor/SZ3.git && cd SZ3
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=.. ..
make
make install
echo "SZ3 installation done."
echo "======================"
cd ../..

# Install SZx
echo "As SZx is not open source, please install it manually by contacting the author."