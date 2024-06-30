FROM python

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip --no-cache-dir install appfl

WORKDIR /APPFL
COPY grpc_mnist_server.py .

CMD ["python3", "grpc_mnist_server.py", "--nclients=1"]