Example: SSL Encrypted gRPC Communication
=========================================

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/3n8a026VqdQ?si=07WxkRgQp5bzmZZq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Generate SSL Certificates and Keys
-----------------------------------

.. note::
    You need to have ``openssl`` installed on your machine to generate the SSL certificates and keys.

In this example, we show how to enable SSL encrypted gRPC communication between the FL server and the clients. APPFL provides a command line interface, ``appfl-setup-ssl``, to generate the SSL certificates and keys for the server and the clients in a user-specified directory. The following block shows the information needed to generate the SSL certificates and keys:

- The absolute path of the directory where the SSL certificate and private key will be stored (``~/.appfl/ssl`` by default).
- Optional country code, state, and organization. Feel free to press Enter to use the default values.
- The DNS and IP address of the server. By default, the DNS is set to ``localhost`` and the IP address is set to ``127.0.0.1``.

.. code-block:: bash

    $ appfl-setup-ssl

    Enter the absolute path of the directory where the SSL certificate and private key will be stored, press Enter to use the default directory /home/.appfl/ssl:
    Enter Country Code, press Enter to use default 'US':
    Enter State, press Enter to use default 'Illinois':
    Enter Organization (O), press Enter to use default 'APPFL':
    Enter DNS (DNS.1), press Enter to use default 'localhost':
    Enter IP, press Enter to use default '127.0.0.1':
    Certificate request self-signature ok
    subject=C=US, ST=Illinois, O=APPFL, CN=localhost
    =============================================================================
    SSL certificate stored in /home/.appfl/ssl/server.crt
    SSL private key stored in /home/.appfl/ssl/server.key
    CA certificate stored in  /home/.appfl/ssl/ca.crt
    Please copy the CA certificate /home/.appfl/ssl/ca.crt to the client machines
    =============================================================================

.. note::
    When you create a server for distributed clients (i.e., not on the same network), you need to provide the DNS name and public IP address of the server.

``appfl-setup-ssl`` generates three files needed for SSL encrypted gRPC communication in the specified directory (``~/.appfl/ssl`` by default):

- ``server.crt``: SSL certificate for the server.
- ``server.key``: SSL private key for the server.
- ``ca.crt``: CA certificate used to verify the server certificate.

.. note::
    The server needs to share the CA certificate with the clients, which should copy it to their own machines.

Server Configuration
--------------------

We use this `server configuration file <https://github.com/APPFL/APPFL/blob/main/examples/resources/configs/mnist/server_fedavg.yaml>`_ as an example to show how to enable SSL encrypted gRPC communication. Modify the ``server_configs.comm_configs.grpc_configs`` field in the server configuration file as follows:

.. code-block:: yaml

    comm_configs:
        grpc_configs:
            server_uri: localhost:50051 # Make sure the server URI corresponds to the IP set in the SSL certificate
            max_message_size: 1048576
            use_ssl: True # Enable SSL encrypted gRPC communication
            server_certificate_key: "/home/.appfl/ssl/server.key" # Path to the server SSL private key
            server_certificate: "/home/.appfl/ssl/server.crt" # Path to the server SSL certificate
            ca_certificate: "/home/.appfl/ssl/ca.crt" # Path to the CA certificate
            # Additional authentication configurations
            use_authenticator: True
            authenticator: "NaiveAuthenticator"
            authenticator_args:
                auth_token: "A_SECRET_DEMO_TOKEN"

As shown in the example configuration above, we also configure token-based authentication. APPFL provides a simple token-based authenticator, ``NaiveAuthenticator``, to authenticate the clients. The server only accepts clients that provide the correct token, which is set in the ``auth_token`` field of ``authenticator_args``.

Client Configuration
--------------------

We use this `client configuration file <https://github.com/APPFL/APPFL/blob/main/examples/resources/configs/mnist/client_1.yaml>`_ as an example to show how to enable SSL encrypted gRPC communication on the client side. Modify the ``comm_configs.grpc_configs`` field in the client configuration file as follows. Note that the ``root_certificate`` field is the path to the ``ca.crt`` file shared by the server, which is used to verify the server's SSL certificate. For the authenticator configurations, the client should provide the same token as the server.

.. code-block:: yaml

    comm_configs:
        grpc_configs:
            server_uri: localhost:50051
            max_message_size: 1048576
            use_ssl: True
            root_certificate: "client_path/ca.crt"
            use_authenticator: True
            authenticator: "NaiveAuthenticator"
            authenticator_args:
                auth_token: "A_SECRET_DEMO_TOKEN"

Run the Server and Clients
--------------------------

After modifying the server and client configuration files, we can run the server and clients as usual, and they will communicate over SSL encrypted gRPC. The above examples use localhost, so you can run the server and two clients on the same machine in three separate terminals.

.. code-block:: bash

    $ cd examples
    $ python grpc/run_server.py --config resources/configs/mnist/server_fedavg.yaml # [Terminal 1]
    $ python grpc/run_client.py --config resources/configs/mnist/client_1.yaml      # [Terminal 2]
    $ python grpc/run_client.py --config resources/configs/mnist/client_2.yaml      # [Terminal 3]
