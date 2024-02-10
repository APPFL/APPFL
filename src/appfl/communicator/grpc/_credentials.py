"""
Loading the testing credentials for the gRPC server and client from files.
Adopted from https://github.com/grpc/grpc/blob/master/examples/python/auth/_credentials.py
"""

import os

def _load_credential_from_file(filepath):
    real_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(real_path, "rb") as f:
        return f.read()

SERVER_CERTIFICATE = _load_credential_from_file("credentials/server_cert.pem")
SERVER_CERTIFICATE_KEY = _load_credential_from_file("credentials/server_key.pem")
ROOT_CERTIFICATE = _load_credential_from_file("credentials/root_cert.pem")
