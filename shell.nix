{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-23.11") {} }:

pkgs.mkShellNoCC {
  packages = with pkgs; [
    (python3.withPackages (ps: [
      ps.pip
      ps.setuptools
      ps.wheel
      ps.tkinter
      ps.plotly
      ps.flask
      ps.numpy
      ps.matplotlib
      ps.pandas
      ps.mpi4py
      ps.torch
    ]))
    mpich
    gcc
    curl
    jq
  ];

shellHook = ''
  # Dynamically locate the correct 64-bit libstdc++.so.6
  LIBSTDCXX_PATH=$(find /nix/store -path "*/lib/libstdc++.so.6.0.32" -exec file {} \; | grep "ELF 64-bit" | awk -F: '{print $1}' | head -n 1)

  # Ensure the library was found
  if [ -n "$LIBSTDCXX_PATH" ]; then
    export LD_LIBRARY_PATH=$(dirname $LIBSTDCXX_PATH):$LD_LIBRARY_PATH
    export LD_PRELOAD=$LIBSTDCXX_PATH
    echo "Using libstdc++.so.6 from $LIBSTDCXX_PATH"
  else
    echo "Error: libstdc++.so.6 not found or no 64-bit version available."
  fi

  # Create a virtual environment if it doesn't exist
  if [ ! -d ".venv" ]; then
    python -m venv .venv
  fi

  # Activate the virtual environment
  source .venv/bin/activate

  # Install appfl with examples if not already installed
  if ! python -c "import appfl" &> /dev/null; then
    pip install "appfl[examples]"
  fi
'';

}

