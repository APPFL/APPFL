# from globus_compute_sdk import Executor

# # from flamby.datasets.fed_heart_disease import FedHeartDisease

# def get_flamby():
#     from flamby.datasets.fed_heart_disease import FedHeartDisease
#     # test_dataset =  FedHeartDisease(train=False, center=0, pooled=False)
#     # train_dataset = FedHeartDisease(train=True, center=0, pooled=False)
#     return "train_dataset, test_dataset"


# tutorial_endpoint_id = 'ae4eb8e4-05d5-4d6e-9ae6-b3c8276d0375'
# with Executor(endpoint_id=tutorial_endpoint_id) as fxe:
#     future = fxe.submit(get_flamby)
#     print(future.result())

from globus_compute_sdk import Executor


# First, define the function ...
def hello_world():
    return "Hello World!"


def add(x):
    return x + 7


tutorial_endpoint_id = (
    "984b0f9e-a2eb-42ec-a0d3-413b3ca92833"  # Public tutorial endpoint
)
# ... then create the executor, ...
with Executor(endpoint_id=tutorial_endpoint_id) as gce:
    # ... then submit for execution, ...
    future = gce.submit(hello_world)

    # ... and finally, wait for the result
    print(future.result())

    future = gce.submit(add, 1)
    print(future.result())
