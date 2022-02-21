On Google Cloud Platform
========================

This describes how to set up ``appfl`` `gRPC <https://grpc.io>`_ server to run on `Google Cloud <https://cloud.google.com>`_ platform, while the clients are training the model locally. In this tutorial, we describe the steps to run a simple example by using the following services:

.. toctree::
    :maxdepth: 1

    endpoints
    cloudrun


.. note::

    We recommend to use Endpoints, as we observed that gRPC server running on Cloud Run occasionally experienced connection issue. See `<https://github.com/APPFL/APPFL/issues/56>`_.


For the tutorials, complete the following steps first.

Creating a Google project
-------------------------

We need to create a `Google project <https://cloud.google.com/resource-manager/docs/creating-managing-projects#gcloud>`_ and set the following environment variable for convenience:

.. code-block:: shell

  export GCP_PROJECT=<YOUR_PROJECT_ID>


Preparing a Docker container
----------------------------

The tutorials on Google Cloud Platform use a Docker container to run a gRPC server. A simple ``Dockerfile`` can be written as follows:

.. literalinclude:: /../examples/gcloud/Dockerfile
    :language: docker
    :linenos:


A Docker container can also be built from the repository:

.. literalinclude:: /../examples/gcloud/Dockerfile.repo
    :language: docker
    :linenos:


Building a Docker container
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We build a docker container by using the tag name ``gcr.io/$GCP_PROJECT/appfl-test:latest``.

.. code-block:: shell

  docker build -t gcr.io/$GCP_PROJECT/appfl-test:latest .


Test the Docker container on local machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the docker image locally:

.. code-block:: shell

  docker run -d -p 50051:50051 -e PORT=50051 gcr.io/$GCP_PROJECT/appfl-test:latest


Once the image running successfully, we can run the ``appfl`` client from the file ``examples/gcloud/grpc_mnist_client.py`` in the repository.

.. code-block:: shell

  python grpc_mnist_client.py --host=localhost --port=50051 --nclients=1 --client_id=1


.. node::

  Make sure to stop the docker run after the test is done.


Deploying the Docker container to the cloud
-------------------------------------------

To deploy the docker container to the Google cloud, we first need to configure ``gcloud``:

.. code-block:: shell

  gcloud auth configure-docker


Now we upload the docker container to the cloud:

.. code-block:: shell

  docker push gcr.io/$GCP_PROJECT/appfl-test:latest