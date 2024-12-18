"""
Loading the testing credentials for the gRPC server and client from files.
Adopted from https://github.com/grpc/grpc/blob/master/examples/python/auth/_credentials.py
"""

import os
from .utils import load_credential_from_file

current_path = os.path.dirname(__file__)
SERVER_CERTIFICATE = load_credential_from_file(
    os.path.join(current_path, "credentials", "localhost.crt")
)
SERVER_CERTIFICATE_KEY = load_credential_from_file(
    os.path.join(current_path, "credentials", "localhost.key")
)
ROOT_CERTIFICATE = load_credential_from_file(
    os.path.join(current_path, "credentials", "root.crt")
)
