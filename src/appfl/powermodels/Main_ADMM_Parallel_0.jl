using PowerModelsDistribution, Ipopt, SCS, JuMP, LinearAlgebra, Printf, Statistics, Distributions, CPLEX, Random
const PMD = PowerModelsDistribution
mutable struct LinDist3Model <: PMD.LPUBFDiagModel PMD.@pmd_fields end
using MPI
include("./Models.jl") 
include("./Structure.jl") 
include("./Functions.jl")
include("./ADMM.jl")

MPI.Init()

global comm = MPI.COMM_WORLD
global rank = MPI.Comm_rank(comm)
global size = MPI.Comm_size(comm)
global root = 0
global status = Array{Int64}([0])


NET         = "IEEE123"
ALGO        = "ADMM_Parallel"
TwoPSS      = "ON"
PROX        = "P1"
ClosedForm  = "ON"
π_init      = 0.995
Initial_ρ   = 10.0
bar_ϵ       = Inf

println("---------start reading data")
info, pm = Read_Info(NET, ALGO, TwoPSS, PROX, ClosedForm, π_init, Initial_ρ, bar_ϵ)
println("---------construct line models")
 

NLines_counts = split_count(info.NLines,size-1)
pushfirst!(NLines_counts,0)

NLines_start, NLines_last = get_start_last(rank, NLines_counts)    
var = Variables() 
Line_info = construct_line_LP_model(pm, info, var)    

line_in_size, line_out_size, line_in_map_idx, line_out_map_idx = calculate_master_line_info(info, Line_info) # for master
Line_in_size, Line_out_size, Line_in_map_idx, Line_out_map_idx = calculate_sub_line_info(Line_info, NLines_start, NLines_last, line_in_size, line_out_size) # for sub


if rank == root        
    res_io = write_output_1(info)      
    println("----solving consensus")       
    optimal = consensus_LP_optimal_points(info, pm)     
    param   = set_initial_points(info, optimal)     
    
    println("Entering ADMM Master: from $(NLines_start) to $(NLines_last)")        
    ADMM_Master(pm, info, Line_info, var, param, NLines_counts, res_io, NLines_start, NLines_last, line_in_size, line_out_size, line_in_map_idx, line_out_map_idx)
    
else        
    println("Entering ADMM Sub: from $(NLines_start) to $(NLines_last)")        
    ADMM_Sub(pm, info, Line_info, var,NLines_start, NLines_last, Line_in_size, Line_out_size, Line_in_map_idx, Line_out_map_idx)    
end

MPI.Barrier(comm)
MPI.Finalize()
