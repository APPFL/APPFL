using PowerModelsDistribution, Ipopt, SCS, JuMP, LinearAlgebra, Printf, Statistics, Distributions, CPLEX, Random

const PMD = PowerModelsDistribution

mutable struct LinDist3Model <: PMD.LPUBFDiagModel PMD.@pmd_fields end

include("./Models.jl") 
include("./Structure.jl") 
include("./Functions.jl")
include("./ADMM.jl")

function run_LPOPF_DG_ADMM_Serial(;NET= "IEEE13", ALGO ="ADMM_Serial", TwoPSS = "ON", PROX = "P1", ClosedForm="ON", π_init=0.995, Initial_ρ=10.0, bar_ϵ=Inf) 
    println("--------------------------START_reading--------------------------")    
    info, pm = Read_Info(NET, ALGO, TwoPSS, PROX, ClosedForm, π_init, Initial_ρ, bar_ϵ)
    println("--------------------------END_reading--------------------------")
    res_io  = write_output_1(info)     
    ## Initial point
    param = consensus_LP_initial_points(info, pm)     
    # param   = set_initial_points(info, optimal)                
    ## Construct models
    var = Variables() 
    Line_info = construct_line_LP_model(pm, info, var)    
    bus_model, biased_bus_model = construct_bus_model(pm, info, var)    
    
    ADMM_Serial(pm, info, param, Line_info, bus_model, biased_bus_model, var, res_io)         
end
 

 




 