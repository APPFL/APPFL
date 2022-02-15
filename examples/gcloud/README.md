# Test on Google Cloud Run

This describes how to set up `appfl` gRPC server to run on [Google Cloud Run](https://cloud.google.com), while the clients are training the model locally.

First we need to set a Google project ID:

```shell
export GCP_PROJECT=grpc-tutorial-123456
```

## Build a docker image

We build a docker container by using the tag name `gcr.io/$GCP_PROJECT/appfl-test:latest`.

```shell
docker build -t gcr.io/$GCP_PROJECT/appfl-test:latest .
```

## Test gRPC with the docker image

The docker image built above can be tested locally or deployed to the cloud.

### On local machine

To run the docker image locally:

```shell
docker run -d -p 50051:50051 -e PORT=50051 gcr.io/$GCP_PROJECT/appfl-test:latest
```

Once the image running successfully, we can run the `appfl` client:

```shell
python grpc_mnist_client.py --host=localhost --port=50051 --nclients=1 --client_id=1
```

> Make sure to stop the docker run after the test is done.


### On Google Cloud Run

To deploy the docker image to the Google cloud, we first need to configure `gcloud`:

```shell
gcloud auth configure-docker
```

Now we upload the docker image to the cloud:

```shell
docker push gcr.io/$GCP_PROJECT/appfl-test:latest
```

The package image can be deployed to the cloud, and the communication port will be open at 50051. The port should be set to the same as in the `appfl` configuration.

```shell
gcloud run deploy appfl-test --image gcr.io/$GCP_PROJECT/appfl-test:latest --platform managed --memory=4G --port=50051
```

Alternatively, additional arguments can also be passed as follows:

```shell
gcloud run deploy appfl-test --image gcr.io/$GCP_PROJECT/appfl-test:latest --platform managed --memory=4G --port=50051 \
--command=python3 \
--args=grpc_mnist_server.py \
--args="--logging=DEBUG" \
--args="--nclients=1"
```

If the package is successfully deployed to the cloud, you may see the messages like this:

```shell
Deploying container to Cloud Run service [appfl-test] in project [grpc-tutorial-123456] region [us-central1]
✓ Deploying new service... Done.
  ✓ Creating Revision...
  ✓ Routing traffic...
  ✓ Setting IAM Policy...
Done.
Service [appfl-test] revision [appfl-test-00001-yif] has been deployed and is serving 100 percent of traffic.
Service URL: https://appfl-test-abcd1234-uc.a.run.app
```

Now the server is ready to listen requests from clients. A client can start the federated learning with the following:

```shell
python grpc_mnist_client.py --host=/appfl-test-abcd1234-uc.a.run.app --use_tls=True --client_id=1 --nclients=3
```

In this example `grpc_mnist_client.py` we pass multiple arguments. The most important one is `--host` and `--use_tls` for the service URL and the indication of using TLS (transport layer security). Note that the port number is not passed.
