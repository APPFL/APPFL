"""
[DEPRECATED]
Warning: This module is deprecated and will be removed in future versions.
Please use:
    - `appfl.aggregator` for server aggregation algorithms,
    - `appfl.trainer` for client local training algorithms,
    - `appfl.scheduler` for server scheduling algorithms 

This module provides a base class to define the functions required to implement PPFL algorithms.
"""
from .fl_base import *
from .ppfl_base import *
from .client_optimizer import *
from .client_step_optimizer import *
from .globus_compute_client_optimizer import *
from .globus_compute_client_step_optimizer import *
from .personalized_client_optimizer import *
from .personalized_client_step_optimizer import *
from .server_fed_avg import *
from .server_fed_avgmom import *
from .server_fed_adagrad import *
from .server_fed_adam import *
from .server_fed_yogi import *
from .iceadmm import *
from .iiadmm import *
from .server_fed_asynchronous import *
from .server_fed_buffer import *
from .server_fed_compass import *
from .server_fed_compass_mom import *
from .server_fed_compass_nova import *
from .scheduler_compass import *
from .scheduler_dummy import *
