#!/bin/bash

run_test() {
    echo "mpiexec -np 2 python cifar10.py --device gpu --server $1 --client_optimizer SGD --num_local_epochs 1 --num_epochs 1000"
}

$(run_test "ServerFedAvg")
$(run_test "ServerFedAvgMomentum")
$(run_test "ServerFedAdagrad")
$(run_test "ServerFedAdam")
$(run_test "ServerFedYogi")
