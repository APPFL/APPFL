#!/usr/bin/env python3
"""
APPFL TES Client Runner

This module provides the entry point for running APPFL client tasks
within GA4GH TES containers, following APPFL architectural patterns.
"""

import sys
import os

# Add the APPFL source to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from appfl.comm.tes.tes_client_communicator import main

if __name__ == "__main__":
    sys.exit(main())