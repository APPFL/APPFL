from cmath import inf
import os
import time
from typing import OrderedDict


from appfl.config import *

import julia
jl = julia.Julia(compiled_modules=False)
from pathlib import Path


import appfl.run as rt
from mpi4py import MPI



DataSet_name = "IEEE13"
ALGO        = "ADMM"
TwoPSS      = "OFF"
PROX        = "OFF"
ClosedForm  = "ON"
Init_sol    = 1.0
Init_rho    = 10.0
bar_eps     = inf


dir = os.getcwd() + "/datasets/RawData"


## Run
def main():
        
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()


    """ Load Julia Functions  """
    st_julia = time.time()
    path_curr = str(Path(os.getcwd()).parent) + "/src/appfl/powermodels/"    
    jl.include(path_curr+"Models.jl")
    jl.include(path_curr+"Structure.jl")
    jl.include(path_curr+"Functions.jl")    
    jl.include(path_curr+"ADMM.jl")    
    time_julia = time.time() - st_julia
    print("time_julia=",time_julia)

    """ Get data """
    info, pm = jl.Read_Info(DataSet_name, ALGO, TwoPSS, PROX, ClosedForm, Init_sol, Init_rho, bar_eps)
    
    res_io  = jl.write_output_1(info)     
    
    """ Initial Points """
    param = jl.consensus_LP_initial_points(info, pm)    

    var = jl.Variables() 
    Line_info = jl.construct_line_LP_model(pm, info, var)    
    bus_model, biased_bus_model = jl.construct_bus_model(pm, info, var)    
    
    jl.ADMM_Serial_TEST(pm, info, param, Line_info, bus_model, biased_bus_model, var, res_io)          
    
    """ Get model """

    
    
    # model = get_model(comm)
    
    """ read default configuration """
    # cfg = OmegaConf.structured(Config)

 
    """ Serial """
    




    # if comm_size > 1:
    #     if comm_rank == 0:
    #         rt.run_server(cfg, comm, model, test_dataset, num_clients, DataSet_name)
    #     else:
    #         rt.run_client(cfg, comm, model, train_datasets, num_clients)
    #     print("------DONE------", comm_rank)
    # else:
    #     rt.run_serial(cfg, model, train_datasets, test_dataset, DataSet_name)


if __name__ == "__main__":
    main()





## python powermodels.py
## mpiexec -np 3 python powermodels.py