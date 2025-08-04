#!/usr/bin/env python3
"""
APPFL TES Client Docker Container Entry Point

This script serves as the entry point for APPFL clients running in TES containers.
It can be used both as a standalone script and as a module import.
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main function from the TES client communicator
try:
    from appfl.comm.tes.tes_client_communicator import main
except ImportError:
    # Fallback for development/testing
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tes_client_communicator", 
        "/app/appfl/comm/tes/tes_client_communicator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = module.main

if __name__ == "__main__":
    main()