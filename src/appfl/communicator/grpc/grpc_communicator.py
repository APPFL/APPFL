from concurrent import futures
import grpc
from grpc_communicator_new_pb2 import Response
from grpc_communicator_new_pb2_grpc import *
from authenticator import APPFLAuthenticator
from appfl.login_manager.globus import GlobusAuthenticator
from appfl.login_manager.naive import NaiveAuthenticator
from _credentials import *

class DictionaryServiceServicer(DictionaryServiceServicer):
    def SendDictionary(self, request, context):
        # Logic to handle the incoming dictionary
        print("Received dictionary:", request.entries)
        return Response(message="Dictionary received successfully.")

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=(
            APPFLAuthenticator(
                GlobusAuthenticator(
                    is_fl_server=True, 
                    globus_group_id="77c1c74b-a33b-11ed-8951-7b5a369c0a53"
                )
            ),
        ),

        # interceptors=(
        #     APPFLAuthenticator(
        #         NaiveAuthenticator()
        #     ),
        # ),
    )
    add_DictionaryServiceServicer_to_server(DictionaryServiceServicer(), server)
    server_credentials = grpc.ssl_server_credentials(
        (
            (
                SERVER_CERTIFICATE_KEY,
                SERVER_CERTIFICATE,
            ),
        )
    )

    server.add_secure_port(
        "localhost:50051", server_credentials
    )
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
