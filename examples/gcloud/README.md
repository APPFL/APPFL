# Test on Google Cloud Run

We need to set a Google project ID:

```shell
export GCP_PROJECT=grpc-tutorial-341102
```

## Build a docker image

```shell
docker build -t gcr.io/$GCP_PROJECT/appfl-test:latest .
```

## Test gRPC with the docker image

### On local machine

Run the docker image:

```shell
docker run -d -p 50051:50051 -e PORT=50051 gcr.io/$GCP_PROJECT/appfl-test:latest
```

And run the `appfl` client:

```shell
python grpc_mnist_client.py --host=localhost --port=50051 --nclients=1 --client_id=1
```

> Make sure to stop the docker run after the test is done.


### On Google Cloud Run

Configure to upload the docker image to Google.

```shell
gcloud auth configure-docker
```

Upload the docker image to the Google Cloud Run

```shell
docker push gcr.io/grpc-tutorial-341102/appfl-test:latest
```

```shell
gcloud run deploy --image gcr.io/grpc-tutorial-341102/appfl-test:latest --platform managed --memory=4G --port=50051
```

If successful,

```shell
Deploying container to Cloud Run service [appfl-test] in project [grpc-tutorial-341102] region [us-central1]
✓ Deploying new service... Done.
  ✓ Creating Revision...
  ✓ Routing traffic...
  ✓ Setting IAM Policy...
Done.
Service [appfl-test] revision [appfl-test-00001-yif] has been deployed and is serving 100 percent of traffic.
Service URL: https://appfl-test-kabyj6s7ma-uc.a.run.app
```