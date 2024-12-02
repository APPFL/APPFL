from funcx import FuncXExecutor

def double(x):
    return x * 2

endpoint_id = '' #YOUR-ENDPOINT-ID
with FuncXExecutor(endpoint_id=endpoint_id) as fxe:
    fut = fxe.submit(double, 7)
    print(fut.result())