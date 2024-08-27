"""
Warning: Importing algorithms directly from this module (e.g., `from appfl.algorithm import ServerFedAvg`)
is deprecated and will be removed in future versions.
Please use:
    - `appfl.algorithm.aggregator` for server aggregation algorithms,
    - `appfl.algorithm.trainer` for client local training algorithms,
    - `appfl.algorithm.scheduler` for server scheduling algorithms 

This module provides a base class to define the functions required to implement PPFL algorithms.
"""

from .legacy.fl_base import *
from .legacy.ppfl_base import *
from .legacy.client_optimizer import *
from .legacy.client_step_optimizer import *
from .legacy.globus_compute_client_optimizer import *
from .legacy.globus_compute_client_step_optimizer import *
from .legacy.personalized_client_optimizer import *
from .legacy.personalized_client_step_optimizer import *
from .legacy.server_fed_avg import *
from .legacy.server_fed_avgmom import *
from .legacy.server_fed_adagrad import *
from .legacy.server_fed_adam import *
from .legacy.server_fed_yogi import *
from .legacy.iceadmm import *
from .legacy.iiadmm import *
from .legacy.server_fed_asynchronous import *
from .legacy.server_fed_buffer import *
from .legacy.server_fed_compass import *
from .legacy.server_fed_compass_mom import *
from .legacy.server_fed_compass_nova import *
from .legacy.scheduler_compass import *
from .legacy.scheduler_dummy import *
