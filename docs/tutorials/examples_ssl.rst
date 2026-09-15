Example: SSL Encrypted gRPC Communication
=========================================

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/3n8a026VqdQ?si=07WxkRgQp5bzmZZq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Generate SSL Certificates and Keys
-----------------------------------

In this example, we show how to enable SSL encrypted gRPC communication between the FL server and the clients. APPFL provides a command line interface, ``appfl-setup-ssl``, to generate the SSL certificates and keys for the server and the clients in the user-specific directory. The generator is implemented directly with the :mod:`cryptography` library (no ``openssl`` binary required).

The following block of code shows what information is needed to generate the SSL certificates and keys:

- The absolute path of the directory where the SSL certificate and private key will be stored, by default they are stored in ``~/.appfl/ssl``.
- Optional country code, state, organization. User can feel free to press Enter to use the default values.
- The DNS name(s) and IP address(es) of the server (comma-separated). By default, the DNS is set to ``localhost`` and the IP address is set to ``127.0.0.1``. A single cert can cover several names — for example ``appfl.example.com, alt.example.com, localhost``.
- A passphrase for the CA private key. The script prompts for it interactively (or reads ``APPFL_CA_PASSPHRASE`` if set). Anyone who reads ``ca.key`` off the disk still needs this passphrase to mint federation-wide certificates.

.. code-block:: bash

    $ appfl-setup-ssl

    Enter the absolute path of the directory where the SSL certificate and private key will be stored, press Enter to use the default directory /home/.appfl/ssl:
    Enter Country Code (2 letters), press Enter to use default 'US':
    Enter State, press Enter to use default 'Illinois':
    Enter Organization (O), press Enter to use default 'APPFL':
    Enter DNS name(s), comma-separated, press Enter to use default 'localhost':
    Enter IP address(es), comma-separated, press Enter to use default '127.0.0.1':
    CA private key passphrase (>= 12 chars, empty = unencrypted): ************
    Confirm passphrase: ************
    ==============================================================================
    CA certificate stored in  /home/.appfl/ssl/ca.crt
    CA private key stored in  /home/.appfl/ssl/ca.key (encrypted)
    Server certificate stored in /home/.appfl/ssl/server.crt
    Server private key stored in /home/.appfl/ssl/server.key
    ==============================================================================
    Copy /home/.appfl/ssl/ca.crt to every client that should trust this federation.
    On each client, set in the gRPC client communicator config:
      root_certificate: /home/.appfl/ssl/ca.crt
      use_ssl: true
    Pin the server identity (must match a SAN below) using:
      server_hostname: localhost
    Subject alternative names on this server cert:
      DNS: localhost
      IP:  127.0.0.1
    ==============================================================================

.. note::
    When you create a server for distributed clients (i.e. not under the same network), you need to provide the DNS and public IP address of the server. They will all be encoded as ``SubjectAlternativeName`` entries on the server certificate. Set ``server_hostname`` on each client to whichever entry that client will use to reach the server.

.. note::
    To run ``appfl-setup-ssl`` non-interactively (CI / containers), set ``APPFL_CA_PASSPHRASE`` in the environment. To deliberately generate an unencrypted CA key (not recommended on shared hosts), set ``APPFL_CA_NO_ENCRYPT=1``.

``appfl-setup-ssl`` will generate three files needed for SSL encrypted gRPC communication in the specified directory (by default, ``~/.appfl/ssl``):

- ``server.crt``: SSL certificate for the server.
- ``server.key``: SSL private key for the server.
- ``ca.crt``: CA certificate for the server.

.. note::
    The server needs to provide the CA certificate to the clients. The clients need to copy the CA certificate to the client machines.

Server Configuration
--------------------

We use this `server configuration file <https://github.com/APPFL/APPFL/blob/main/examples/resources/configs/mnist/server_fedavg.yaml>`_ as an example to show how to modify the server configuration file to enable SSL encrypted gRPC communication. We need to modify the ``server_configs.comm_configs.grpc_configs`` field in the server configuration file to enable SSL encrypted gRPC communication as the following:

.. code-block:: yaml

    comm_configs:
        grpc_configs:
            server_uri: localhost:50051 # Make sure the server URI corresponds to the IP set in the SSL certificate
            max_message_size: 1048576
            use_ssl: True # Enable SSL encrypted gRPC communication
            server_certificate_key: "/home/.appfl/ssl/server.key" # Path to the server SSL private key
            server_certificate: "/home/.appfl/ssl/server.crt" # Path to the server SSL certificate
            ca_certificate: "/home.appfl/ssl/ca.crt" # Path to the CA certificate
            # Additional authentication configurations
            use_authenticator: True
            authenticator: "NaiveAuthenticator"
            authenticator_args:
                auth_token: "A_SECRET_DEMO_TOKEN"

As shown in the example configuration above, we also provide additional token-based authentication configurations. APPFL provides a simple token-based authenticator, ``NaiveAuthenticator``, to authenticate the clients. The server will only accept the clients that provide the correct token. The token is set in the ``auth_token`` field in the ``authenticator_args`` field.

Client Configuration
--------------------

We use this `client configuration file <https://github.com/APPFL/APPFL/blob/main/examples/resources/configs/mnist/client_1.yaml>`_ as an example to show how to modify the client configuration file to enable SSL encrypted gRPC communication. We need to modify the ``comm_configs.grpc_configs`` field in the client configuration file to enable SSL encrypted gRPC communication as the following. It should be noted that the ``root_certificate`` field is the path to the ``ca.crt`` file shared by the server to verify the server's SSL certificate. As for the authenticator configurations, the client should provide the same token as the server.

.. code-block:: yaml

    comm_configs:
        grpc_configs:
            server_uri: localhost:50051
            max_message_size: 1048576
            use_ssl: True
            root_certificate: "client_path/ca.crt"
            server_hostname: localhost  # must match a SAN on the server cert
            use_authenticator: True
            authenticator: "NaiveAuthenticator"
            authenticator_args:
                auth_token: "A_SECRET_DEMO_TOKEN"

.. note::
    ``server_hostname`` is required when ``use_ssl: True``. It is the identity
    the client expects the server to prove, independent of ``server_uri``.
    Without it, any certificate that chains to the configured CA would be
    accepted regardless of subject — an attacker who can mint a leaf cert from
    the same CA could MITM the connection.

Run the Server and Clients
--------------------------

After modifying the server and client configuration files, we can run the server and clients as usual. The server and clients will establish SSL encrypted gRPC communication. The above examples use localhost, so you can run the server and two clients on the same machine within three separate terminals.

.. code-block:: bash

    $ cd examples
    $ python grpc/run_server.py --config resources/configs/mnist/server_fedavg.yaml # [Terminal 1]
    $ python grpc/run_client.py --config resources/configs/mnist/client_1.yaml      # [Terminal 2]
    $ python grpc/run_client.py --config resources/configs/mnist/client_2.yaml      # [Terminal 3]
