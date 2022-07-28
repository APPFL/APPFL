def get_model():
    import torchvision
    import torch.nn as nn
    class DenseNet121(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        """
        def __init__(self, num_output):
            super(DenseNet121, self).__init__()
            self.densenet121 = torchvision.models.densenet121(pretrained = False)
            num_features = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_features, num_output),
                nn.Sigmoid()
            )
        def forward(self, x):
            x = self.densenet121(x)
            return x
    ## User-defined model
    return DenseNet121


from parsl.addresses import address_by_hostname
from parsl.launchers import SingleNodeLauncher
from parsl.providers import CobaltProvider

from funcx_endpoint.endpoint.utils.config import Config
from funcx_endpoint.executors import HighThroughputExecutor
from funcx_endpoint.strategies import SimpleStrategy

# fmt: off
# PLEASE UPDATE user_opts BEFORE USE
user_opts = {
    'theta': {
        'worker_init': '. /home/thoang/miniconda3/etc/profile.d/conda.sh; conda activate appfl',
        'scheduler_options': '',
        # Specify the account/allocation to which jobs should be charged
        'account': 'covid-ct'
    }
}

config = Config(
    executors=[
        HighThroughputExecutor(
            strategy=SimpleStrategy(max_idletime=300),
            worker_debug=False,
            address="10.236.1.193", #thetalogin4
            provider=CobaltProvider(
                queue='single-gpu',
                account=user_opts['theta']['account'],
                launcher=SingleNodeLauncher(),

                # string to prepend to #COBALT blocks in the submit
                # script to the scheduler eg: '#COBALT -t 50'
                scheduler_options=user_opts['theta']['scheduler_options'],

                # Command to be run before starting a worker, such as:
                # 'module load Anaconda; source activate funcx_env'.
                worker_init=user_opts['theta']['worker_init'],

                # Scale between 0-1 blocks with 2 nodes per block
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,

                # Hold blocks for 30 minutes
                walltime='00:30:00'
            ),
        )
    ],
)

# fmt: on
# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": "thetagpu-cli-1",
    "description": "",
    "organization": "",
    "department": "",
    "public": False,
    "visible_to": [],
}