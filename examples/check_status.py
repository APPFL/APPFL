from appfl.funcx import check_endpoint
from funcx import FuncXClient

fxc = FuncXClient()

check_endpoint(
    fxc,
    ["f45229d0-487d-4981-b062-f20b0ba4fd95", "b425e356-2bb2-449e-a3dc-bf7cdfcca379"],
)
