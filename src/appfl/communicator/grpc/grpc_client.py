import grpc
from .grpc_communicator_new_pb2 import *
from .grpc_communicator_new_pb2_grpc import *
from ._credentials import ROOT_CERTIFICATE
from .authenticator import APPFLAuthMetadataProvider
from appfl.login_manager.globus import GlobusAuthenticator

import contextlib

@contextlib.contextmanager
def create_client_channel(addr):
    # Call credential object will be invoked for every single RPC
    call_credentials = grpc.metadata_call_credentials(
        APPFLAuthMetadataProvider(GlobusAuthenticator(is_fl_server=False))
    )
    # call_credentials = grpc.metadata_call_credentials(
    #     APPFLAuthMetadataProvider(NaiveAuthenticator())
    # )
    # Channel credential will be valid for the entire channel
    channel_credential = grpc.ssl_channel_credentials(
        ROOT_CERTIFICATE
    )
    # Combining channel credentials and call credentials together
    composite_credentials = grpc.composite_channel_credentials(
        channel_credential,
        call_credentials,
    )
    channel = grpc.secure_channel(addr, composite_credentials)
    yield channel

def run():
    with create_client_channel('localhost:50051') as channel:
        stub = DictionaryServiceStub(channel)
        dictionary = {"key1": "value1", "key2": "value2"}
        print(dictionary)
        response = stub.SendDictionary(StringDictionary(entries=dictionary))
        print("Server response:", response.message)

if __name__ == '__main__':
    run()
