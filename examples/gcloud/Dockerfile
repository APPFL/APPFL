FROM ubuntu

RUN apt-get update
RUN apt-get install git python3 python3-pip -yq

RUN git clone -b kkim/grpc https://github.com/APPFL/APPFL.git

WORKDIR /APPFL
COPY ../grpc_mnist_server.py mnist_test_data.pickle .

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip --no-cache-dir install .

CMD ["python3", "grpc_mnist_server.py", "--nclients=1"]
