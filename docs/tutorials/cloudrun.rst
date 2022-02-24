How to run on Cloud Run
=======================

This describes how to set up the Dockerized gRPC server to run on `Google Cloud Run <https://cloud.google.com/run>`_, while the clients are training the model locally.
We assume that the Docker container ``gcr.io/$GCP_PROJECT/appfl-test:latest`` is already available. If you are not sure, see :ref:`Preparing a Docker container`.


Deploying the Dockerized gRPC server
------------------------------------

The Docker container can be deployed to the cloud, and the communication port will be open at 50051. The port should be set to the same as in the `appfl` configuration.

.. code-block:: shell

  gcloud beta run deploy appfl-test \
  --execution-environment gen2 \
  --image gcr.io/$GCP_PROJECT/appfl-test:latest \
  --platform managed \
  --memory=4G \
  --port=50051 \
  --command=python3 \
  --args=grpc_mnist_server.py \
  --args="--logging=INFO" \
  --args="--nclients=3"


The last four arguments replace the default ``CMD`` in ``Dockerfile``, and thus may have values of your choice.
If the package is successfully deployed to the cloud, you may see the messages like this:

.. code-block:: shell

  Deploying container to Cloud Run service [appfl-test] in project [grpc-tutorial-123456] region [us-central1]
  ✓ Deploying new service... Done.
    ✓ Creating Revision...
    ✓ Routing traffic...
    ✓ Setting IAM Policy...
  Done.
  Service [appfl-test] revision [appfl-test-00001-yif] has been deployed and is serving 100 percent of traffic.
  Service URL: https://appfl-test-abcd1234-uc.a.run.app


Now the server is ready to listen requests from clients. The service URL is given in the last line from the above: ``https://appfl-test-abcd1234-uc.a.run.app``, and this needs to be used as ``host`` when launching clients. 


Launching a client
------------------

A client can start the federated learning with the following:

.. code-block:: shell

  python grpc_mnist_client.py \
  --host=/appfl-test-abcd1234-uc.a.run.app \
  --use_tls=True \
  --client_id=1 \
  --nclients=3


.. note::

  In this example ``grpc_mnist_client.py`` we pass multiple arguments. The most important one is ``--host`` and ``--use_tls`` for the service URL and the indication of using TLS (transport layer security). Note that the port number is not passed.
