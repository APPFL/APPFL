import numpy as np

error_bounds = [
    0.25,
    0.2,
    0.15,
    0.1,
    0.09,
    0.08,
    0.07,
    0.06,
    0.05,
    0.04,
    0.03,
    0.02,
    0.01,
]

num_client = 9

num_epochs = 50

server_algorithms = [
    "ServerFedAvg",
    "ServerFedAvgMomentum",
    "ServerFedYogi",
    "ServerFedAdagrad",
    "ServerFedAdam",
]

for server_algorithm in server_algorithms:
    for error_bound in error_bounds:
        print(
            "mpiexec -np %d python3 ./mnist.py --server %s --error_bound %f --num_clients %d --num_epochs %d"
            % (num_client + 1, server_algorithm, error_bound, num_client, num_epochs)
        )
        print(
            "mpiexec -np %d python3 ./mnist.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client"
            % (num_client + 1, server_algorithm, error_bound, num_client, num_epochs)
        )
        print(
            "mpiexec -np %d python3 ./mnist.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --compressed_server"
            % (num_client + 1, server_algorithm, error_bound, num_client, num_epochs)
        )
