using PowerModelsDistribution, Printf, Base, CPLEX
const PMD = PowerModelsDistribution
mutable struct LinDist3Model <: PMD.LPUBFDiagModel PMD.@pmd_fields end


function Initialize(pm::LinDist3Model)  end

function Read(NET)
    FolderName = "./Datasets/RawData/" 
    if NET== "IEEE13"        
        InputFileName = "$(FolderName)13Bus/IEEE13Nodeckt.dss"
    end
    if NET == "IEEE34"        
        InputFileName = "$(FolderName)34Bus/ieee34Mod2.dss"
    end
    if NET == "IEEE37"        
        InputFileName = "$(FolderName)37Bus/ieee37.dss"
    end
    if NET == "IEEE123"    
        InputFileName = "$(FolderName)123Bus/Run_IEEE123Bus.DSS"   ## IEEE123Master.dss  Run_IEEE123Bus.dss  IEEE123Switches
    end

    eng = PMD.parse_file(InputFileName)    
    math = PMD.transform_data_model(eng)  
    return math
end

function Read_Info(NET, ALGO, TwoPSS, PROX, ClosedForm, π_init, Initial_ρ, bar_ϵ)

    math = Read(NET)    
    pm = PMD.instantiate_mc_model(math, LinDist3Model, Initialize)     

    ## Generate Info
    info = Info()
    info.Instance   = NET    
    info.ALGO       = ALGO
    info.TwoPSS     = TwoPSS
    info.PROX       = PROX        
    info.ClosedForm = ClosedForm
    info.π_init     = π_init
    info.Initial_ρ  = Initial_ρ 
    info.bar_ϵ      = bar_ϵ    

    info = generate_information(pm,info)  
      
    return info, pm
end

function generate_information(pm,info;nw=0)

    info.map_bus_to_gen  = Dict( gen["gen_bus"]   => i for (i,gen) in PMD.ref(pm, nw, :gen))
    info.map_bus_to_load = Dict( load["load_bus"] => Vector{Int64}()  for (i,load) in PMD.ref(pm, nw, :load) )        
    [ push!(info.map_bus_to_load[load["load_bus"]], i) for (i,load) in PMD.ref(pm, nw, :load) ]
    
    ## Wye bus
    info.Wye_bus = [];
    load_wye_ids = [id for (id, load) in PMD.ref(pm, nw, :load) if load["configuration"]==WYE]    
    for id ∈ load_wye_ids
        load = PMD.ref(pm, nw, :load, id)   
        if load["load_bus"] ∉ info.Wye_bus
            push!(info.Wye_bus, load["load_bus"])
        end        
    end     

    ## Delta bus
    info.Delta_bus = [];
    load_del_ids = [id for (id, load) in PMD.ref(pm, nw, :load) if load["configuration"]==DELTA]
    for id ∈ load_del_ids
        load = PMD.ref(pm, nw, :load, id)   
        if load["load_bus"] ∉ info.Delta_bus
            push!(info.Delta_bus, load["load_bus"])
        end        
    end 
    ## Delta bus with one positive load
    info.Special_Delta_bus = [];    
    for i in info.Delta_bus
        tmpcnt = 0
        if length(info.map_bus_to_load[i]) == 1
            for id ∈ info.map_bus_to_load[i]             
                load = PMD.ref(pm, nw, :load, id)                            
                pd0 = load["pd"]
                for idx = 1:length(pd0)
                    if pd0[idx] == 0.0
                        tmpcnt = 1
                    end
                end
            end
        end
        if tmpcnt == 1 && i ∉ info.Special_Delta_bus
            push!(info.Special_Delta_bus, i)                            
        end
    end      

    ## map_bus_idx
    tmpcnt = 0
    for (i,bus) in PMD.ref(pm, :bus)  
        for c in bus["terminals"]
            tmpcnt += 1
            info.map_bus_idx[i,c] = tmpcnt    
        end
    end
    ## map_gen_idx
    tmpcnt = 0
    for (i,gen) in PMD.ref(pm, :gen)  
        for c in gen["connections"]
            tmpcnt += 1
            info.map_gen_idx[i,c] = tmpcnt
        end
    end
    ## map_line_idx
    tmpcnt = 0    
    for (l,transformer) in PMD.ref(pm, :transformer)  
        i1 = transformer["f_bus"]
        i2 = transformer["t_bus"]
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)
        for c in transformer["f_connections"]
            tmpcnt += 1
            info.map_line_idx[f_idx,c] = tmpcnt        
            tmpcnt += 1
            info.map_line_idx[t_idx,c] = tmpcnt                    
        end
    end
    for (l,branch) in PMD.ref(pm, :branch)  
        i1 = branch["f_bus"]
        i2 = branch["t_bus"]
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)
        for c in branch["f_connections"]
            tmpcnt += 1
            info.map_line_idx[f_idx,c] = tmpcnt        
            tmpcnt += 1
            info.map_line_idx[t_idx,c] = tmpcnt       
        end
    end

    ## Construct setG and setA    
    for (i,bus) in PMD.ref(pm, :bus)          
        for c in bus["terminals"]            
            info.setG[i,c]=[]; info.setA[i,c]=[]
            if i in keys(info.map_bus_to_gen) 
                gen_id = info.map_bus_to_gen[i]       
                push!(info.setG[i,c], gen_id)
            end
            if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)    
                    if c in connections    
                        push!(info.setA[i,c], (l,i1,i2) )                        
                    end
                end
            end
            if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)  
                    if c in connections    
                        push!(info.setA[i,c], (l,i1,i2) )                        
                    end
                end
            end
        end
    end 

    ##
    [info.ngen += 1 for (i,gen) in PMD.ref(pm, :gen)  for c in gen["connections"]]
    [info.ntrans += 1 for (i,transformer) in PMD.ref(pm, :transformer)  for c in transformer["f_connections"]]
    [info.nbranch += 1 for (i,branch) in PMD.ref(pm, :branch)  for c in branch["f_connections"]]
    [info.nbus += 1 for (i,bus) in PMD.ref(pm, :bus)  for c in bus["terminals"]]         

    [info.NLines += 1 for (i,transformer) in PMD.ref(pm, :transformer)]
    [info.NLines += 1 for (i,branch) in PMD.ref(pm, :branch)]
    [info.NBuses += 1 for (i,bus) in PMD.ref(pm, :bus)]

    return info
end

########################################################
#### Consensus Optimal Points and Initial Points
########################################################

function consensus_LP_initial_points(info, pm; nw=0)    
    param = ADMMParameter(info) 
    pm, var = construct_initial_consensus_LP_model(pm, info)        
    optimizer = JuMP.optimizer_with_attributes(CPLEX.Optimizer)
    JuMP.set_optimizer(pm.model, optimizer)      
    JuMP.set_silent(pm.model)   
    JuMP.optimize!(pm.model)
    # println("Status=", JuMP.termination_status(pm.model))
    # println("OBJ=", JuMP.objective_value(pm.model))
    
    for (i,gen) in PMD.ref(pm, nw, :gen)        
        for c in gen["connections"]
            idx = info.map_gen_idx[i,c]
            param.pg[idx] = JuMP.value(var.pg[i,c])
            param.qg[idx] = JuMP.value(var.qg[i,c])
            
            param.λ_pg[idx] = -dual(constraint_by_name(pm.model, "CS_pg_[$(i),$(c)]"))
            param.λ_qg[idx] = -dual(constraint_by_name(pm.model, "CS_qg_[$(i),$(c)]"))
        end
    end    
    for (l,branch) in PMD.ref(pm, :branch)        
        i1 = branch["f_bus"]
        i2 = branch["t_bus"]                    
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)
        for c in branch["f_connections"]              
            line_f_idx = info.map_line_idx[f_idx,c]
            line_t_idx = info.map_line_idx[t_idx,c]                
            param.p[line_f_idx] = JuMP.value( var.p[f_idx,c]  )
            param.p[line_t_idx] = JuMP.value( var.p[t_idx,c]  )                
            param.q[line_f_idx] = JuMP.value( var.q[f_idx,c]  )
            param.q[line_t_idx] = JuMP.value( var.q[t_idx,c]  )
            param.w_hat[line_f_idx] = JuMP.value( var.w_hat[f_idx,c] )
            param.w_hat[line_t_idx] = JuMP.value( var.w_hat[t_idx,c] )

            param.λ_p[line_f_idx] = -dual(constraint_by_name(pm.model, "CS_p_[$(f_idx),$(c)]"))
            param.λ_q[line_f_idx] = -dual(constraint_by_name(pm.model, "CS_q_[$(f_idx),$(c)]"))
            param.λ_w[line_f_idx] = -dual(constraint_by_name(pm.model, "CS_w_[$(f_idx),$(c)]"))
            
            param.λ_p[line_t_idx] = -dual(constraint_by_name(pm.model, "CS_p_[$(t_idx),$(c)]"))
            param.λ_q[line_t_idx] = -dual(constraint_by_name(pm.model, "CS_q_[$(t_idx),$(c)]"))
            param.λ_w[line_t_idx] = -dual(constraint_by_name(pm.model, "CS_w_[$(t_idx),$(c)]"))
        end
    end
    for (l,transformer) in PMD.ref(pm, :transformer)        
        i1 = transformer["f_bus"]
        i2 = transformer["t_bus"]                    
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)
        for c in transformer["f_connections"]              
            line_f_idx = info.map_line_idx[f_idx,c]
            line_t_idx = info.map_line_idx[t_idx,c]                
            param.p[line_f_idx] = JuMP.value( var.p[f_idx,c]  )
            param.p[line_t_idx] = JuMP.value( var.p[t_idx,c]  )                
            param.q[line_f_idx] = JuMP.value( var.q[f_idx,c]  )
            param.q[line_t_idx] = JuMP.value( var.q[t_idx,c]  )
            param.w_hat[line_f_idx] = JuMP.value( var.w_hat[f_idx,c] )
            param.w_hat[line_t_idx] = JuMP.value( var.w_hat[t_idx,c] )

            param.λ_p[line_f_idx] = -dual(constraint_by_name(pm.model, "CS_p_[$(f_idx),$(c)]"))
            param.λ_q[line_f_idx] = -dual(constraint_by_name(pm.model, "CS_q_[$(f_idx),$(c)]"))
            param.λ_w[line_f_idx] = -dual(constraint_by_name(pm.model, "CS_w_[$(f_idx),$(c)]"))
            
            param.λ_p[line_t_idx] = -dual(constraint_by_name(pm.model, "CS_p_[$(t_idx),$(c)]"))
            param.λ_q[line_t_idx] = -dual(constraint_by_name(pm.model, "CS_q_[$(t_idx),$(c)]"))
            param.λ_w[line_t_idx] = -dual(constraint_by_name(pm.model, "CS_w_[$(t_idx),$(c)]"))
        end
    end
    for (i,bus) in PMD.ref(pm, nw, :bus)                
        
        for c in bus["terminals"]
            bus_idx = info.map_bus_idx[i,c]
            param.w[bus_idx] = JuMP.value( var.w[i,c] )
            
            if i in keys(info.map_bus_to_gen)
                gen_id = info.map_bus_to_gen[i]
                gen = PMD.ref(pm, nw, :gen, gen_id)                            
                if c in gen["connections"]       
                    idx = info.map_gen_idx[gen_id,c]

                    param.pg_hat[idx] = JuMP.value( var.pg_hat[gen_id,c]  )
                    param.qg_hat[idx] = JuMP.value( var.qg_hat[gen_id,c]  )
                end
            end 
            if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)                                         
                    if c in connections    
                        idx = info.map_line_idx[(l,i1,i2),c]

                        param.p_hat[idx] = JuMP.value( var.p_hat[(l,i1,i2),c] )
                        param.q_hat[idx] = JuMP.value( var.q_hat[(l,i1,i2),c] )                        
                    end
                end
            end
            if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)            
                    if c in connections                                                                         
                        idx = info.map_line_idx[(l,i1,i2),c]

                        param.p_hat[idx] = JuMP.value( var.p_hat[(l,i1,i2),c] )
                        param.q_hat[idx] = JuMP.value( var.q_hat[(l,i1,i2),c] )                        
                    end
                end
            end
        end                    
    end


    ###
    param.ρ_pg    .= info.Initial_ρ
    param.ρ_qg    .= info.Initial_ρ
    param.ρ_p     .= info.Initial_ρ 
    param.ρ_q     .= info.Initial_ρ 
    param.ρ_w     .= info.Initial_ρ 
    return param
end 
  
function load_expmodel_params(load::Dict, bus::Dict, pd, qd)
    # pd = load["pd"]
    # qd = load["qd"]
    
    ncnds = length(pd)
    if load["model"]==POWER
        return (pd, zeros(ncnds), qd, zeros(ncnds))
    else
        # get exponents
        if load["model"]==CURRENT
            alpha = ones(ncnds)
            beta  =ones(ncnds)
        elseif load["model"]==IMPEDANCE
            alpha = ones(ncnds)*2
            beta  =ones(ncnds)*2
        elseif load["model"]==EXPONENTIAL
            alpha = load["alpha"]
            @assert(all(alpha.>=0))
            beta = load["beta"]
            @assert(all(beta.>=0))
        else
            error("load model '$(load["model"])' on 'load.$(load["name"])' unrecongized")
        end
        # calculate proportionality constants
        v0 = load["vnom_kv"]
        a = pd./v0.^alpha
        b = qd./v0.^beta
        # get bounds
        return (a, alpha, b, beta)
    end
end

########################################################
#### Print and Write
########################################################

function write_output_1(info)    
    
    Directory = "Outputs_$(info.Instance)"
    if Base.Filesystem.isdir(Directory) == false
        Base.Filesystem.mkdir(Directory)
    end
 
    rho = @sprintf("%.E", info.Initial_ρ)
    epsabs = @sprintf("%.E", info.ϵ_abs)
    epsrel = @sprintf("%.E", info.ϵ_rel)      
    
    path = "./Outputs_$(info.Instance)/$(info.Instance)_$(info.ALGO)_2PSS_$(info.TwoPSS)_Prox_$(info.PROX)_Closed_$(info.ClosedForm)_rho_$(rho)_pi_$(info.π_init)_epsabs_$(epsabs)_epsrel_$(epsrel)_bareps_$(info.bar_ϵ)"
    outfile = get_new_filename(path, ".txt")
    res_io = open(outfile*".txt", "w")
    return res_io
end

function get_new_filename(prefix::AbstractString, ext::AbstractString)
    outfile = prefix
    if isfile(outfile*ext)
        num = 1
        numcopy = @sprintf("_%d", num)
        while isfile(outfile*numcopy*ext)
            num += 1
            numcopy = @sprintf("_%d", num)
        end
        outfile = outfile*numcopy
    end
    return outfile
end

function print_iteration_title(res_io)
   
    
    @printf(res_io,
        "%12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  \n",
        "Iter",      
        "ρ_mean",              
        "OBJ",             
        "Prim_res", 
        "Dual_res", 
        "ϵ_prim",
        "ϵ_dual",
        "Δ_avg",
        "Elapsed[s]",
        "Gen[s]",
        "Line[s]",
        "Bus[s]",
        "Dual[s]",
        "Iter[s]"        
        )    
    flush(res_io)                              
    @printf(
        "%12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s \n",
        "Iter",      
        "ρ_mean",       
        "OBJ",             
        "Prim_res", 
        "Dual_res", 
        "ϵ_prim",
        "ϵ_dual",
        "Δ_avg",
        "Elapsed[s]",
        "Gen[s]",
        "Line[s]",
        "Bus[s]",
        "Dual[s]",
        "Iter[s]"
        )            

end

function print_iteration(res_io, pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, Δ_avg, elapsed_time, gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)
  
    @printf(res_io,
    "%12d  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.2f  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e \n",
    iteration,  
    ρ_mean,                 
    100.0* sum(pg_hat),             
    primal_res, 
    dual_res, 
    ϵ_prim,
    ϵ_dual, 
    Δ_avg,
    elapsed_time,
    gen_elapsed,
    line_elapsed, 
    bus_elapsed, 
    dual_elapsed, 
    iter_elapsed        
    )      
    flush(res_io)                          
    @printf(
        "%12d  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.2f  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  \n",
        iteration,  
        ρ_mean,                 
        100.0* sum(pg_hat),             
        primal_res, 
        dual_res, 
        ϵ_prim,
        ϵ_dual,
        Δ_avg,
        elapsed_time,
        gen_elapsed,
        line_elapsed, 
        bus_elapsed, 
        dual_elapsed, 
        iter_elapsed               
        )    

 
end

function print_summary(pg_hat, bar_ϵ, res_io) 
    @printf("\n")                    
    @printf("OBJ        . . . . . . . . . . % 12.6e \n", 100.0* sum(pg_hat))
    @printf("Pg         . . . . . . . . . . % 12s \n", round.(pg_hat,digits=4))
    @printf("bareps     . . . . . . . . . . % 12.2f \n", bar_ϵ)
    
    @printf(res_io, "\n")        
    @printf(res_io, "OBJ        . . . . . . . . . . % 12.6e \n", 100.0* sum(pg_hat))        
    @printf(res_io, "Pg         . . . . . . . . . . % 12s \n", round.(pg_hat,digits=4))
    @printf(res_io, "bareps     . . . . . . . . . . % 12.2f \n", bar_ϵ)
    close(res_io)    
end

########################################################
#### Parallel
########################################################
function split_count(N::Integer, n::Integer)
    q,r = divrem(N,n)
    return [i<=r ? q+1 : q for i=1:n]
end

function get_start_last(rank, N_counts)
    
    start = 1 + sum(N_counts[1:rank+1]) -  sum(N_counts[rank+1])
    last = sum(N_counts[1:rank+1])

    return start, last
end

function calculate_master_line_info(info, Line_info) 
    N_in = 18
    N_out = 6

    line_in_size=Dict();  line_in_map_idx = Dict();
    line_out_size=Dict(); line_out_map_idx = Dict();
    
    tmp_in = 1;  tmp_out = 1;  
    for line_idx = 1: info.NLines        
            
        line_in_size[line_idx] = 0
        line_out_size[line_idx] = 0
        
        for c in Line_info[line_idx][6]    

            line_in_size[line_idx]  += N_in
            line_out_size[line_idx] += N_out

            for j = 1:N_in
                line_in_map_idx[line_idx, c, j] = tmp_in
                tmp_in += 1
            end
            for j = 1:N_out
                line_out_map_idx[line_idx, c, j] = tmp_out
                tmp_out += 1
            end
        end
    end 

    return line_in_size, line_out_size, line_in_map_idx, line_out_map_idx
end

function calculate_sub_line_info(Line_info, NLines_start, NLines_last, line_in_size, line_out_size)
    N_in = 18
    N_out = 6

    Line_in_size = 0;  Line_in_map_idx = Dict()
    Line_out_size = 0; Line_out_map_idx = Dict()    

    tmp_in = 1;  tmp_out = 1;  
    for line_idx = NLines_start:NLines_last
        
        Line_in_size  += line_in_size[line_idx]       
        Line_out_size += line_out_size[line_idx]       

        for c in Line_info[line_idx][6]

            for j = 1:N_in
                Line_in_map_idx[line_idx, c, j] = tmp_in
                tmp_in += 1
            end
            for j = 1:N_out
                Line_out_map_idx[line_idx, c, j] = tmp_out
                tmp_out += 1
            end
        end        
    end
    return Line_in_size, Line_out_size, Line_in_map_idx, Line_out_map_idx
end

function compute_Lines_counts(NLines_counts, line_in_size, line_out_size)

    Temp_ACC_NC = [0]
    
    Lines_in_counts = Vector{Int64}();  Lines_out_counts = Vector{Int64}();
    tmp = 0
    for i = 1:length(NLines_counts)        
        tmp += NLines_counts[i]
        push!(Temp_ACC_NC, tmp) 
    end
  
     
    for i = 1:length(Temp_ACC_NC)-1
        tmp_in = 0; tmp_out = 0
        for j = Temp_ACC_NC[i] + 1: Temp_ACC_NC[i+1]
            tmp_in  += line_in_size[j]
            tmp_out += line_out_size[j]
        end
        push!(Lines_in_counts, tmp_in)
        push!(Lines_out_counts, tmp_out)
    end

    return Lines_in_counts, Lines_out_counts
end   

function construct_Lines_vector(info, param, Line_info) 

    Lines_in_vector = Vector{Float64}()
    Lines_out_vector = Vector{Float64}()
    for line_idx = 1: info.NLines
        
        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]   
        
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)
 
        for c in connections
            line_f_idx = info.map_line_idx[f_idx,c]
            line_t_idx = info.map_line_idx[t_idx,c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]
            
            ## Inputs
            # ρ_p_f = param.ρ_p[line_f_idx]
            # ρ_p_t = param.ρ_p[line_t_idx]
            # ρ_q_f = param.ρ_q[line_f_idx]
            # ρ_q_t = param.ρ_q[line_t_idx]
            # ρ_w_f = param.ρ_w[line_f_idx]
            # ρ_w_t = param.ρ_w[line_t_idx]
            # λ_p_f = param.λ_p[line_f_idx]
            # λ_p_t = param.λ_p[line_t_idx]
            # λ_q_f = param.λ_q[line_f_idx]
            # λ_q_t = param.λ_q[line_t_idx]
            # λ_w_f = param.λ_w[line_f_idx]
            # λ_w_t = param.λ_w[line_t_idx]
            # p_hat_f = param.p_hat[line_f_idx]
            # p_hat_t = param.p_hat[line_t_idx]
            # q_hat_f = param.q_hat[line_f_idx]
            # q_hat_t = param.q_hat[line_t_idx]
            # w_f     = param.w[bus_f_idx]
            # w_t     = param.w[bus_t_idx]
            ## Outputs
            # p_f = param.p[line_f_idx]
            # p_t = param.p[line_t_idx]
            # q_f = param.q[line_f_idx]
            # q_t = param.q[line_t_idx]
            # w_hat_f = param.w_hat[line_f_idx]
            # w_hat_t = param.w_hat[line_t_idx]
            
            # push!(Lines_in_vector, 
            #         ρ_p_f, ρ_p_t, ρ_q_f, ρ_q_t, ρ_w_f, ρ_w_t, 
            #         λ_p_f, λ_p_t, λ_q_f, λ_q_t, λ_w_f, λ_w_t,
            #         p_hat_f, p_hat_t, q_hat_f, q_hat_t, w_f, w_t)
                                       
            # push!(Lines_out_vector, p_f, p_t, q_f, q_t, w_hat_f, w_hat_t) 

            push!(Lines_in_vector, param.ρ_p[line_f_idx],param.ρ_p[line_t_idx],param.ρ_q[line_f_idx],param.ρ_q[line_t_idx],param.ρ_w[line_f_idx],param.ρ_w[line_t_idx], param.λ_p[line_f_idx],param.λ_p[line_t_idx],param.λ_q[line_f_idx],param.λ_q[line_t_idx],param.λ_w[line_f_idx],param.λ_w[line_t_idx], param.p_hat[line_f_idx],param.p_hat[line_t_idx],param.q_hat[line_f_idx],param.q_hat[line_t_idx],param.w[bus_f_idx],param.w[bus_t_idx])
                                       
            push!(Lines_out_vector, param.p[line_f_idx],param.p[line_t_idx],param.q[line_f_idx],param.q[line_t_idx],param.w_hat[line_f_idx],param.w_hat[line_t_idx]) 

        end        
    end 

    return Lines_in_vector, Lines_out_vector
end