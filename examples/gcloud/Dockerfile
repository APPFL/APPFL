FROM ubuntu

RUN apt-get update
RUN apt-get install git python3 python3-pip -yq

RUN git clone -b kkim/grpc https://github.com/APPFL/APPFL.git

WORKDIR /APPFL
COPY grpc_mnist_server.py mnist_test_data.pickle .

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip --no-cache-dir install .
# RUN python3 -m grpc_tools.protoc \
#         -I. \
#         --python_out=. \
#         --grpc_python_out=. \
#         federated_learning.proto

CMD ["python3", "grpc_mnist_server.py", "--host=localhost", "--nclients=1"]


# docker build -t gcr.io/grpc-tutorial-341102/appfl-test:latest .
# gcloud auth configure-docker
# docker push gcr.io/grpc-tutorial-341102/appfl-test:latest
# gcloud run deploy --image gcr.io/grpc-tutorial-341102/appfl-test:latest --platform managed --memory=4G --port=50051

# ENDPOINT=$(\
#   gcloud run services list \
#   --project=grpc-tutorial-341102 \
#   --region=us-central1 \
#   --platform=managed \
#   --format="value(status.address.url)" \
#   --filter="metadata.name=appfl-test") 
# ENDPOINT=${ENDPOINT#https://} && echo ${ENDPOINT}