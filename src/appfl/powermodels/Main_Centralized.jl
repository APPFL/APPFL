using PowerModelsDistribution, Ipopt, CPLEX, SCS, JuMP, LinearAlgebra, Printf
mutable struct LinDist3Model <: PMD.LPUBFDiagModel PMD.@pmd_fields end
const PMD = PowerModelsDistribution

include("./Models.jl") 
include("./Functions.jl")
include("./Structure.jl") 
 
function run_LPOPF_DG_centralized(;NET= "IEEE13",Approach="PMD")
      
    math = Read(NET)    

    if Approach == "PMD" 
        start_time = time() 
        pm = PMD.instantiate_mc_model(math, PMD.LPUBFDiagPowerModel, build_mc_opf)
        result = optimize_model!(pm, optimizer=Ipopt.Optimizer)    
        println( result["solution"]["gen"]["1"]["pg"] )        
        println( "Objective=", result["objective"]  )
        elapsed_time = time() -start_time
        println("elapsed_time=", elapsed_time)
    end

    if Approach == "Consensus"
        pm = PMD.instantiate_mc_model(math, LinDist3Model, Initialize)        
        info = Info() 
        info = generate_information(pm,info)  
        pm, var = construct_consensus_LP_model(pm, info)
        optimizer = JuMP.optimizer_with_attributes(CPLEX.Optimizer)
        JuMP.set_optimizer(pm.model, optimizer)        
        JuMP.optimize!(pm.model)        
        println("Status=", JuMP.termination_status(pm.model))
        println("OBJ=", 100*JuMP.objective_value(pm.model))
        for (i,gen) in PMD.ref(pm, 0, :gen)        
            for c in gen["connections"]
                println("var.pg[$(i),$(c)]=",JuMP.value(var.pg[i,c]))                
            end
        end            
    end
 
end




 