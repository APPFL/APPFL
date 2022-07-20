from parsl.providers import LocalProvider

from funcx_endpoint.endpoint.utils.config import Config
from funcx_endpoint.executors import HighThroughputExecutor
from parsl.channels import LocalChannel
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider

from parsl.addresses import address_by_hostname, address_by_route
user_opts = {
    'bebop': {
        'worker_init': '. /home/thoang/miniconda3/etc/profile.d/conda.sh; conda activate appfl',
        'scheduler_options': '',
    }
}

config = Config(
    executors=[
        HighThroughputExecutor(
            cores_per_worker=1,
            worker_debug=False,
            address= address_by_hostname(),
            provider=SlurmProvider(
                partition='knls',
                channel=LocalChannel(),
                launcher=SrunLauncher(),
                scheduler_options=user_opts['bebop']['scheduler_options'],
                worker_init=user_opts['bebop']['worker_init'],

                # Scale between 0-1 blocks with 2 nodes per block
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,

                # Hold blocks for 30 minutes
                walltime='00:30:00'
            ),
        )]
)

# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": <ENDPOINT NAME>,
    "description": "",
    "organization": "",
    "department": "",
    "public": False,
    "visible_to": [],
}
