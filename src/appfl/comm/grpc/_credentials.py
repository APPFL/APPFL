"""
Loading the testing credentials for the gRPC server and client from files.
Adopted from https://github.com/grpc/grpc/blob/master/examples/python/auth/_credentials.py
"""

import os

def load_credential_from_file(filepath):
    with open(filepath, "rb") as f:
        return f.read()

current_path = os.path.dirname(__file__)
SERVER_CERTIFICATE = load_credential_from_file(
    os.path.join(current_path, "credentials/localhost.crt")
)
SERVER_CERTIFICATE_KEY = load_credential_from_file(
    os.path.join(current_path, "credentials/localhost.key")
)
ROOT_CERTIFICATE = load_credential_from_file(
    os.path.join(current_path, "credentials/root.crt")
)
