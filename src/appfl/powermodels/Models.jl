using JuMP, LinearAlgebra

########################################################
#### component-based decomposed LP models (Note: branch and transformer models are LP)
########################################################
function construct_branch_LP_model(pm, info, branch, l,  model, var; nw=0)
    JuMP.@objective(model, Min, 0)
    i1 = branch["f_bus"]
    i2 = branch["t_bus"]
    f_idx = (l,i1,i2)
    t_idx = (l,i2,i1)

    smax_fr = PMD._calc_branch_power_max(PMD.ref(pm, nw, :branch, l), PMD.ref(pm, nw, :bus, i1))
    smax_to = PMD._calc_branch_power_max(PMD.ref(pm, nw, :branch, l), PMD.ref(pm, nw, :bus, i2))
    
    bus_fr = PMD.ref(pm, nw, :bus, i1)               
    bus_to = PMD.ref(pm, nw, :bus, i2)   

    r = branch["br_r"]
    x = branch["br_x"]
    g_sh_fr = branch["g_fr"]
    g_sh_to = branch["g_to"]
    b_sh_fr = branch["b_fr"]
    b_sh_to = branch["b_to"]

    angmin = branch["angmin"]
    angmax = branch["angmax"]
    
    f_connections = branch["f_connections"]
    t_connections = branch["t_connections"]

    N = length(f_connections)

    alpha = exp(-im*2*pi/3)
    Gamma = [1 alpha^2 alpha; alpha 1 alpha^2; alpha^2 alpha 1][f_connections,t_connections]

    MP = 2*(real(Gamma).*r + imag(Gamma).*x)
    MQ = 2*(real(Gamma).*x - imag(Gamma).*r)
    
    for c in branch["f_connections"]
        line_f_idx = info.map_line_idx[f_idx,c]
        line_t_idx = info.map_line_idx[t_idx,c]

        var.p[f_idx,c] = JuMP.@variable(model,  base_name="p_$(f_idx)[$(c)]" )
        var.p[t_idx,c] = JuMP.@variable(model,  base_name="p_$(t_idx)[$(c)]" )

        var.q[f_idx,c] = JuMP.@variable(model,  base_name="q_$(f_idx)[$(c)]" )
        var.q[t_idx,c] = JuMP.@variable(model,  base_name="q_$(t_idx)[$(c)]" )  
        
        var.w_hat[f_idx,c] = JuMP.@variable(model,  base_name="w_hat_$(f_idx)[$(c)]", lower_bound=0.0 )
        var.w_hat[t_idx,c] = JuMP.@variable(model,  base_name="w_hat_$(t_idx)[$(c)]", lower_bound=0.0 )
    end

    # for (idx, c) in enumerate(info.branch_connections[f_idx])
    for (idx, c) in enumerate(f_connections)        
        PMD.set_upper_bound(var.p[f_idx,c],  smax_fr[idx])
        PMD.set_lower_bound(var.p[f_idx,c],  -smax_fr[idx])
        PMD.set_upper_bound(var.q[f_idx,c],  smax_fr[idx])
        PMD.set_lower_bound(var.q[f_idx,c],  -smax_fr[idx])    
    end

    # for (idx, c) in enumerate(info.branch_connections[t_idx])
    for (idx, c) in enumerate(t_connections)  
        PMD.set_upper_bound(var.p[t_idx,c],  smax_to[idx])
        PMD.set_lower_bound(var.p[t_idx,c],  -smax_to[idx])
        PMD.set_upper_bound(var.q[t_idx,c],  smax_to[idx])
        PMD.set_lower_bound(var.q[t_idx,c],  -smax_to[idx])
    end

    for (idx, t) in enumerate(bus_fr["terminals"])
        if t in branch["f_connections"]
            PMD.set_upper_bound(var.w_hat[f_idx,t], max(bus_fr["vmin"][idx]^2, bus_fr["vmax"][idx]^2))        
            if bus_fr["vmin"][idx] > 0
                PMD.set_lower_bound(var.w_hat[f_idx,t], bus_fr["vmin"][idx]^2)
            end
        end
    end  
    for (idx, t) in enumerate(bus_to["terminals"])
        if t in branch["f_connections"]
            PMD.set_upper_bound(var.w_hat[t_idx,t], max(bus_to["vmin"][idx]^2, bus_to["vmax"][idx]^2))        
            if bus_fr["vmin"][idx] > 0
                PMD.set_lower_bound(var.w_hat[t_idx,t], bus_to["vmin"][idx]^2)
            end
        end
    end  
    ## Constraints
    p_s_fr = [var.p[f_idx,fc] - diag(g_sh_fr)[idx].*var.w_hat[f_idx,fc] for (idx,fc) in enumerate(f_connections)]
    q_s_fr = [var.q[f_idx,fc] + diag(b_sh_fr)[idx].*var.w_hat[f_idx,fc] for (idx,fc) in enumerate(f_connections)]
    
    for (idx, (fc,tc)) in enumerate(zip(f_connections, t_connections))        
        ## Loss
        JuMP.@constraint(model, var.p[f_idx,fc] + var.p[t_idx,tc] == g_sh_fr[idx,idx]*var.w_hat[f_idx,fc] +  g_sh_to[idx,idx]*var.w_hat[t_idx,tc])
        JuMP.@constraint(model, var.q[f_idx,fc] + var.q[t_idx,tc] == -b_sh_fr[idx,idx]*var.w_hat[f_idx,fc] -b_sh_to[idx,idx]*var.w_hat[t_idx,tc])

        ## Magnitude difference                
        JuMP.@constraint(model, var.w_hat[t_idx,tc] == var.w_hat[f_idx,fc]  - sum(MP[idx,j]*p_s_fr[j] for j in 1:N) - sum(MQ[idx,j]*q_s_fr[j] for j in 1:N))
        ## Angle difference
        g_fr = branch["g_fr"][idx,idx]
        g_to = branch["g_to"][idx,idx]
        b_fr = branch["b_fr"][idx,idx]
        b_to = branch["b_to"][idx,idx]
        r = branch["br_r"][idx,idx]
        x = branch["br_x"][idx,idx]
        JuMP.@constraint(model,
            tan(angmin[idx])*((1 + r*g_fr - x*b_fr)*(var.w_hat[f_idx,fc]) - r*var.p[f_idx,fc] - x*var.q[f_idx,fc])
                     <= ((-x*g_fr - r*b_fr)*(var.w_hat[f_idx,fc]) + x*var.p[f_idx,fc] - r*var.q[f_idx,fc])
            )
        JuMP.@constraint(model,
            tan(angmax[idx])*((1 + r*g_fr - x*b_fr)*(var.w_hat[f_idx,fc]) - r*var.p[f_idx,fc] - x*var.q[f_idx,fc])
                     >= ((-x*g_fr - r*b_fr)*(var.w_hat[f_idx,fc]) + x*var.p[f_idx,fc] - r*var.q[f_idx,fc])
            )
    end
    
end

function construct_trans_LP_model(pm, info, transformer, l,  model, var; nw=0, fix_taps::Bool=true)
    JuMP.@objective(model, Min, 0)
    i1 = transformer["f_bus"]
    i2 = transformer["t_bus"]
    f_idx = (l,i1,i2)
    t_idx = (l,i2,i1)
    configuration = transformer["configuration"]        
    f_connections = transformer["f_connections"]
    t_connections = transformer["t_connections"]
    tm_set = transformer["tm_set"]
    nph = length(tm_set)
    tm_fixed = fix_taps ? ones(Bool, length(tm_set)) : transformer["tm_fix"]
    tm_scale = PMD.calculate_tm_scale(transformer, PMD.ref(pm, nw, :bus, i1), PMD.ref(pm, nw, :bus, i2))
    pol = transformer["polarity"]
    tm = [tm_fixed[idx] ? tm_set[idx] : var(pm, nw, :tap, trans_id)[fc] for (idx,(fc,tc)) in enumerate(zip(f_connections,t_connections))]

    bus_fr = PMD.ref(pm, nw, :bus, i1)               
    bus_to = PMD.ref(pm, nw, :bus, i2) 

    Pt_f_idx=Dict(); Qt_f_idx=Dict(); Pt_t_idx=Dict(); Qt_t_idx=Dict()

    # println(info.trans_connections[f_idx])
    # println(f_connections)
    # stop

    (Pt_f_idx, Qt_f_idx) = PMD.variable_mx_complex(model, [f_idx], Dict(f_idx=>f_connections) , Dict(f_idx=>f_connections); name=("Pt", "Qt"))   
    (Pt_t_idx, Qt_t_idx) = PMD.variable_mx_complex(model, [t_idx], Dict(t_idx=>t_connections) , Dict(t_idx=>t_connections); name=("Pt", "Qt"))   

    # (Pt_f_idx, Qt_f_idx) = PMD.variable_mx_complex(model, [f_idx], Dict(f_idx=>info.trans_connections[f_idx]) , Dict(f_idx=>info.trans_connections[f_idx]); name=("Pt", "Qt"))   
    # (Pt_t_idx, Qt_t_idx) = PMD.variable_mx_complex(model, [t_idx], Dict(t_idx=>info.trans_connections[t_idx]) , Dict(t_idx=>info.trans_connections[t_idx]); name=("Pt", "Qt"))   

    for c in transformer["f_connections"]
        line_f_idx = info.map_line_idx[f_idx,c]
        line_t_idx = info.map_line_idx[t_idx,c]

        var.p[f_idx,c] = diag(Pt_f_idx[f_idx])[c]                 
        var.p[t_idx,c] = diag(Pt_t_idx[t_idx])[c]
        
        var.q[f_idx,c] = diag(Qt_f_idx[f_idx])[c]   
        var.q[t_idx,c] = diag(Qt_t_idx[t_idx])[c]        
        var.w_hat[f_idx,c] = JuMP.@variable(model,  base_name="w_hat_$(f_idx)[$(c)]", lower_bound=0.0 )
        var.w_hat[t_idx,c] = JuMP.@variable(model,  base_name="w_hat_$(t_idx)[$(c)]", lower_bound=0.0 )
               
     end

    for (idx, t) in enumerate(bus_fr["terminals"])
        if t in transformer["f_connections"]
            PMD.set_upper_bound(var.w_hat[f_idx,t], max(bus_fr["vmin"][idx]^2, bus_fr["vmax"][idx]^2))        
            if bus_fr["vmin"][idx] > 0
                PMD.set_lower_bound(var.w_hat[f_idx,t], bus_fr["vmin"][idx]^2)
            end
        end
    end  
    for (idx, t) in enumerate(bus_to["terminals"])
        if t in transformer["f_connections"]
            PMD.set_upper_bound(var.w_hat[t_idx,t], max(bus_to["vmin"][idx]^2, bus_to["vmax"][idx]^2))        
            if bus_fr["vmin"][idx] > 0
                PMD.set_lower_bound(var.w_hat[t_idx,t], bus_to["vmin"][idx]^2)
            end
        end
    end  

    if configuration == WYE
        for c in transformer["f_connections"]                
            JuMP.@constraint(model, var.p[f_idx,c] + var.p[t_idx,c] == 0)
            JuMP.@constraint(model, var.q[f_idx,c] + var.q[t_idx,c] == 0)    
        end
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))          
            JuMP.@constraint(model, var.w_hat[f_idx,fc] == (pol*tm_scale*tm[idx])^2*var.w_hat[t_idx,tc])
        end    
        
    elseif configuration == DELTA
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))          
            jdx = (idx-1+1)%nph+1
            fd = f_connections[jdx]                        
            JuMP.@constraint(model, 3.0*(var.w_hat[f_idx,fc] + var.w_hat[f_idx,fd]) == 2.0*(pol*tm_scale*tm[idx])^2*var.w_hat[t_idx,tc])
        end    
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))          
            jdx = (idx-1+nph-1)%nph+1
            fd = f_connections[jdx]
            td = t_connections[jdx]

            JuMP.@constraint(model, 2*var.p[f_idx,fc] == -(var.p[t_idx,tc]+var.p[t_idx,td])+(var.q[t_idx,td]-var.q[t_idx,tc])/sqrt(3.0) )
            JuMP.@constraint(model, 2*var.q[f_idx,fc] == (var.p[t_idx,tc]-var.p[t_idx,td])/sqrt(3.0) - (var.q[t_idx,td]+var.q[t_idx,tc]) )
        end  
    end
end

function construct_gen_model(pm, info, gen, i,  model, var)    
    bus_id = gen["gen_bus"]
    for c in gen["connections"]     
        idx = info.map_gen_idx[i,c]                      
        var.pg[i,c] = JuMP.@variable(model, base_name="pg_$(i)[$(c)]", lower_bound=gen["pmin"][c], upper_bound=gen["pmax"][c])
        var.qg[i,c] = JuMP.@variable(model, base_name="qg_$(i)[$(c)]", lower_bound=gen["qmin"][c], upper_bound=gen["qmax"][c]) 
    end        

    OBJ = 0        
    [OBJ += var.pg[i,c] for c in gen["connections"]]                                  
    JuMP.@objective(model, Min, OBJ)            
end

function construct_bus_model(pm, info, bus, i,  model, var; nw=0)
    JuMP.@objective(model, Min, 0)
 
    terminals = bus["terminals"]
    grounded =  bus["grounded"]
    for c in bus["terminals"]
        idx = info.map_bus_idx[i,c]                
        var.w[i,c] = JuMP.@variable(model, base_name="w_$(i)[$(c)]")
    end
    ## generator bus
    if i in keys(info.map_bus_to_gen) 
        gen_id = info.map_bus_to_gen[i]       
        gen = PMD.ref(pm, nw, :gen, gen_id)                    
        for c in gen["connections"]                         
            idx = info.map_gen_idx[gen_id,c]
            var.pg_hat[gen_id,c] = JuMP.@variable(model, base_name="pg_hat_$(gen_id)[$(c)]", lower_bound=gen["pmin"][c], upper_bound=gen["pmax"][c] )
            var.qg_hat[gen_id,c] = JuMP.@variable(model, base_name="qg_hat_$(gen_id)[$(c)]", lower_bound=gen["qmin"][c], upper_bound=gen["qmax"][c] ) 
        end
    end

    ## load bus
    if i in keys(info.map_bus_to_load)        
        for id in info.map_bus_to_load[i]   
            load = PMD.ref(pm, nw, :load, id)                            

            pd0 = load["pd"]
            qd0 = load["qd"]   
 
            a, alpha, b, beta = PMD._load_expmodel_params(load, bus)
            vmin, vmax = PMD._calc_load_vbounds(load, bus)            
            pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
            connections = load["connections"]  
             
            if load["configuration"] == DELTA
                (Xdr,Xdi) = PMD.variable_mx_complex(model, [id], Dict(id => connections), Dict(id => connections); name="Xd")
                
                Td = [1 -1 0; 0 1 -1; -1 0 1]   
                pd_bus = LinearAlgebra.diag(Xdr[id]*Td)
                qd_bus = LinearAlgebra.diag(Xdi[id]*Td)
                pd = LinearAlgebra.diag(Td*Xdr[id])
                qd = LinearAlgebra.diag(Td*Xdi[id])

                # println("id=", id, " connections=", connections, " pd0=", pd0, " Xdr=", Xdr, " pd_bus=",pd_bus, " pd=", pd)
 
                for (idx, _) in enumerate(connections)                    
                    var.pd[id,idx] = pd_bus[idx]
                    var.qd[id,idx] = qd_bus[idx]
                end

                for (idx, _) in enumerate(connections)                    
                    if abs(pd0[idx]+im*qd0[idx]) == 0.0
                        JuMP.@constraint(model, Xdr[id][:,idx] .== 0)
                        JuMP.@constraint(model, Xdi[id][:,idx] .== 0)
                    end
                end 

                ## 3 different load types
                if load["model"]==POWER
                    for (idx, _) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==pd0[idx])
                        JuMP.@constraint(model, qd[idx]==qd0[idx])
                    end
                elseif load["model"]==IMPEDANCE                    
                    for (idx,_) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==3*a[idx]*var.w[i,idx])
                        JuMP.@constraint(model, qd[idx]==3*b[idx]*var.w[i,idx])
                    end
                elseif load["model"]==CURRENT 
                    for (idx,_) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==sqrt(3)/2*a[idx]*(var.w[i,idx]+1))
                        JuMP.@constraint(model, qd[idx]==sqrt(3)/2*b[idx]*(var.w[i,idx]+1))
                    end
                end
 
            end 

            if load["configuration"] == WYE                                
                for (idx,c) in enumerate(connections)  
                    if load["model"]==POWER                                                        
                        var.pd[id,c] = pd0[idx]
                        var.qd[id,c] = qd0[idx]            
                        
                    elseif load["model"]==IMPEDANCE                                                            
                        var.pd[id,c] = a[idx] * var.w[i,c]
                        var.qd[id,c] = b[idx] * var.w[i,c]    
                        
                    elseif load["model"]==CURRENT                         
                        var.pd[id,c] = JuMP.@variable(model, base_name="pd_$(id)[$(c)]")
                        var.qd[id,c] = JuMP.@variable(model, base_name="qd_$(id)[$(c)]")                         
        
                        PMD.set_lower_bound(var.pd[id,c], pmin[idx])
                        PMD.set_upper_bound(var.pd[id,c], pmax[idx])
                        PMD.set_lower_bound(var.qd[id,c], qmin[idx])
                        PMD.set_upper_bound(var.qd[id,c], qmax[idx])
                        
                        JuMP.@constraint(model, var.pd[id,c]==1/2*a[idx]*(var.w[i,c]+1))
                        JuMP.@constraint(model, var.qd[id,c]==1/2*b[idx]*(var.w[i,c]+1))                        
                    end  
                end
            end             
        end
    end     
    
    if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
        Pt=Dict(); Qt=Dict()
        for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)                     
            (Pt, Qt) = PMD.variable_mx_complex(model, [(l,i1,i2)], Dict((l,i1,i2)=>connections) , Dict((l,i1,i2)=>connections); name=("Pt_hat", "Qt_hat"))
            for c in connections
                idx = info.map_line_idx[(l,i1,i2),c]
                var.p_hat[(l,i1,i2),c] = diag(Pt[(l,i1,i2)])[c]
                var.q_hat[(l,i1,i2),c] = diag(Qt[(l,i1,i2)])[c]                      
            end            
        end             
    end
    if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
        for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)  
            for c in connections
                idx = info.map_line_idx[(l,i1,i2),c]
                var.p_hat[(l,i1,i2),c] = JuMP.@variable(model, base_name="p_hat_$((l,i1,i2))[$(c)]")
                var.q_hat[(l,i1,i2),c] = JuMP.@variable(model, base_name="q_hat_$((l,i1,i2))[$(c)]")
            end
        end
    end
      
    bus_arcs = PMD.ref(pm, nw, :bus_arcs_conns_branch, i)
    bus_arcs_sw = PMD.ref(pm, nw, :bus_arcs_conns_switch, i)
    bus_arcs_trans = PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)
    bus_gens = PMD.ref(pm, nw, :bus_conns_gen, i)
    bus_storage = PMD.ref(pm, nw, :bus_conns_storage, i)
    bus_loads = PMD.ref(pm, nw, :bus_conns_load, i)
    bus_shunts = PMD.ref(pm, nw, :bus_conns_shunt, i)
    
    ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]
    
    for (idx,t) in ungrounded_terminals    
        
        JuMP.@constraint(model,
          sum( var.p_hat[a,t] for (a, conns) in bus_arcs if t in conns)        
        + sum( var.p_hat[a,t] for (a, conns) in bus_arcs_trans if t in conns)
        - sum( var.pg_hat[g,t] for (g, conns) in bus_gens if t in conns)        
        + sum( var.pd[d,t] for (d, conns) in bus_loads if t in conns)
        + sum(diag(PMD.ref(pm, nw, :shunt, sh, "gs"))[findfirst(isequal(t), conns)]*var.w[i,t] for (sh, conns) in bus_shunts if t in conns)
        == 0.0, base_name="pb1[$(t)]" )
        JuMP.@constraint(model,
              sum( var.q_hat[a,t] for (a, conns) in bus_arcs if t in conns)            
            + sum( var.q_hat[a,t] for (a, conns) in bus_arcs_trans if t in conns)
            - sum( var.qg_hat[g,t] for (g, conns) in bus_gens if t in conns)            
            + sum( var.qd[d,t] for (d, conns) in bus_loads if t in conns)
            - sum(diag(PMD.ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(t), conns)]*var.w[i,t] for (sh, conns) in bus_shunts if t in conns)
            == 0.0, base_name="pb2[$(t)]")
    end
end

function construct_initial_bus_model(pm, info, bus, i,  model, var; nw=0)
    JuMP.@objective(model, Min, 0)
 
    terminals = bus["terminals"]
    grounded =  bus["grounded"]
    for c in bus["terminals"]
        idx = info.map_bus_idx[i,c]                
        var.w[i,c] = JuMP.@variable(model, base_name="w_$(i)[$(c)]")
    end
    ## generator bus
    if i in keys(info.map_bus_to_gen) 
        gen_id = info.map_bus_to_gen[i]       
        gen = PMD.ref(pm, nw, :gen, gen_id)                    
        for c in gen["connections"]                         
            idx = info.map_gen_idx[gen_id,c]
            var.pg_hat[gen_id,c] = JuMP.@variable(model, base_name="pg_hat_$(gen_id)[$(c)]", lower_bound=gen["pmin"][c], upper_bound=gen["pmax"][c] )
            var.qg_hat[gen_id,c] = JuMP.@variable(model, base_name="qg_hat_$(gen_id)[$(c)]", lower_bound=gen["qmin"][c], upper_bound=gen["qmax"][c] ) 
        end
    end

    ## load bus
    if i in keys(info.map_bus_to_load)        
        for id in info.map_bus_to_load[i]   
            load = PMD.ref(pm, nw, :load, id)                            

            pd0 = load["pd"].*info.π_init
            qd0 = load["qd"]   
 
            a, alpha, b, beta = PMD._load_expmodel_params(load, bus)
            vmin, vmax = PMD._calc_load_vbounds(load, bus)            
            pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
            connections = load["connections"]  
             
            if load["configuration"] == DELTA
                (Xdr,Xdi) = PMD.variable_mx_complex(model, [id], Dict(id => connections), Dict(id => connections); name="Xd")
                
                Td = [1 -1 0; 0 1 -1; -1 0 1]   
                pd_bus = LinearAlgebra.diag(Xdr[id]*Td)
                qd_bus = LinearAlgebra.diag(Xdi[id]*Td)
                pd = LinearAlgebra.diag(Td*Xdr[id])
                qd = LinearAlgebra.diag(Td*Xdi[id])

                # println("id=", id, " connections=", connections, " pd0=", pd0, " Xdr=", Xdr, " pd_bus=",pd_bus, " pd=", pd)
 
                for (idx, _) in enumerate(connections)                    
                    var.pd[id,idx] = pd_bus[idx]
                    var.qd[id,idx] = qd_bus[idx]
                end

                for (idx, _) in enumerate(connections)                    
                    if abs(pd0[idx]+im*qd0[idx]) == 0.0
                        JuMP.@constraint(model, Xdr[id][:,idx] .== 0)
                        JuMP.@constraint(model, Xdi[id][:,idx] .== 0)
                    end
                end 

                ## 3 different load types
                if load["model"]==POWER
                    for (idx, _) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==pd0[idx])
                        JuMP.@constraint(model, qd[idx]==qd0[idx])
                    end
                elseif load["model"]==IMPEDANCE                    
                    for (idx,_) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==3*a[idx]*var.w[i,idx])
                        JuMP.@constraint(model, qd[idx]==3*b[idx]*var.w[i,idx])
                    end
                elseif load["model"]==CURRENT 
                    for (idx,_) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==sqrt(3)/2*a[idx]*(var.w[i,idx]+1))
                        JuMP.@constraint(model, qd[idx]==sqrt(3)/2*b[idx]*(var.w[i,idx]+1))
                    end
                end
 
            end 

            if load["configuration"] == WYE                                
                for (idx,c) in enumerate(connections)  
                    if load["model"]==POWER                                                        
                        var.pd[id,c] = pd0[idx]
                        var.qd[id,c] = qd0[idx]            
                        
                    elseif load["model"]==IMPEDANCE                                                            
                        var.pd[id,c] = a[idx] * var.w[i,c]
                        var.qd[id,c] = b[idx] * var.w[i,c]    
                        
                    elseif load["model"]==CURRENT                         
                        var.pd[id,c] = JuMP.@variable(model, base_name="pd_$(id)[$(c)]")
                        var.qd[id,c] = JuMP.@variable(model, base_name="qd_$(id)[$(c)]")                         
        
                        PMD.set_lower_bound(var.pd[id,c], pmin[idx])
                        PMD.set_upper_bound(var.pd[id,c], pmax[idx])
                        PMD.set_lower_bound(var.qd[id,c], qmin[idx])
                        PMD.set_upper_bound(var.qd[id,c], qmax[idx])
                        
                        JuMP.@constraint(model, var.pd[id,c]==1/2*a[idx]*(var.w[i,c]+1))
                        JuMP.@constraint(model, var.qd[id,c]==1/2*b[idx]*(var.w[i,c]+1))                        
                    end  
                end
            end             
        end
    end     
    
    if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
        Pt=Dict(); Qt=Dict()
        for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)                     
            (Pt, Qt) = PMD.variable_mx_complex(model, [(l,i1,i2)], Dict((l,i1,i2)=>connections) , Dict((l,i1,i2)=>connections); name=("Pt_hat", "Qt_hat"))
            for c in connections
                idx = info.map_line_idx[(l,i1,i2),c]
                var.p_hat[(l,i1,i2),c] = diag(Pt[(l,i1,i2)])[c]
                var.q_hat[(l,i1,i2),c] = diag(Qt[(l,i1,i2)])[c]                      
            end            
        end             
    end
    if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
        for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)  
            for c in connections
                idx = info.map_line_idx[(l,i1,i2),c]
                var.p_hat[(l,i1,i2),c] = JuMP.@variable(model, base_name="p_hat_$((l,i1,i2))[$(c)]")
                var.q_hat[(l,i1,i2),c] = JuMP.@variable(model, base_name="q_hat_$((l,i1,i2))[$(c)]")
            end
        end
    end
      
    bus_arcs = PMD.ref(pm, nw, :bus_arcs_conns_branch, i)
    bus_arcs_sw = PMD.ref(pm, nw, :bus_arcs_conns_switch, i)
    bus_arcs_trans = PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)
    bus_gens = PMD.ref(pm, nw, :bus_conns_gen, i)
    bus_storage = PMD.ref(pm, nw, :bus_conns_storage, i)
    bus_loads = PMD.ref(pm, nw, :bus_conns_load, i)
    bus_shunts = PMD.ref(pm, nw, :bus_conns_shunt, i)
    
    ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]
    
    for (idx,t) in ungrounded_terminals    
        
        JuMP.@constraint(model,
          sum( var.p_hat[a,t] for (a, conns) in bus_arcs if t in conns)        
        + sum( var.p_hat[a,t] for (a, conns) in bus_arcs_trans if t in conns)
        - sum( var.pg_hat[g,t] for (g, conns) in bus_gens if t in conns)        
        + sum( var.pd[d,t] for (d, conns) in bus_loads if t in conns)
        + sum(diag(PMD.ref(pm, nw, :shunt, sh, "gs"))[findfirst(isequal(t), conns)]*var.w[i,t] for (sh, conns) in bus_shunts if t in conns)
        == 0.0, base_name="pb1[$(t)]" )
        JuMP.@constraint(model,
              sum( var.q_hat[a,t] for (a, conns) in bus_arcs if t in conns)            
            + sum( var.q_hat[a,t] for (a, conns) in bus_arcs_trans if t in conns)
            - sum( var.qg_hat[g,t] for (g, conns) in bus_gens if t in conns)            
            + sum( var.qd[d,t] for (d, conns) in bus_loads if t in conns)
            - sum(diag(PMD.ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(t), conns)]*var.w[i,t] for (sh, conns) in bus_shunts if t in conns)
            == 0.0, base_name="pb2[$(t)]")
    end
end

function construct_biased_bus_model(pm, info, bus, i,  model, var; nw=0)
    JuMP.@objective(model, Min, 0)
 
    terminals = bus["terminals"]
    grounded =  bus["grounded"]
    for c in bus["terminals"]
        idx = info.map_bus_idx[i,c]                
        var.biased_w[i,c] = JuMP.@variable(model, base_name="w_$(i)[$(c)]")
    end
    ## generator bus
    if i in keys(info.map_bus_to_gen) 
        gen_id = info.map_bus_to_gen[i]       
        gen = PMD.ref(pm, nw, :gen, gen_id)                    
        for c in gen["connections"]                         
            idx = info.map_gen_idx[gen_id,c]
            var.biased_pg_hat[gen_id,c] = JuMP.@variable(model, base_name="pg_hat_$(gen_id)[$(c)]", lower_bound=gen["pmin"][c], upper_bound=gen["pmax"][c] )
            var.biased_qg_hat[gen_id,c] = JuMP.@variable(model, base_name="qg_hat_$(gen_id)[$(c)]", lower_bound=gen["qmin"][c], upper_bound=gen["qmax"][c] ) 
        end
    end

    ## load bus
    if i in keys(info.map_bus_to_load)        
        for id in info.map_bus_to_load[i]   
            load = PMD.ref(pm, nw, :load, id)                            

            pd0 = load["pd"].*info.β
            qd0 = load["qd"]   
 
            a, alpha, b, beta = load_expmodel_params(load, bus, pd0, qd0)
            vmin, vmax = PMD._calc_load_vbounds(load, bus)            
            pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
            connections = load["connections"]  
             
            if load["configuration"] == DELTA
                (Xdr,Xdi) = PMD.variable_mx_complex(model, [id], Dict(id => connections), Dict(id => connections); name="Xd")
                
                Td = [1 -1 0; 0 1 -1; -1 0 1]   
                pd_bus = LinearAlgebra.diag(Xdr[id]*Td)
                qd_bus = LinearAlgebra.diag(Xdi[id]*Td)
                pd = LinearAlgebra.diag(Td*Xdr[id])
                qd = LinearAlgebra.diag(Td*Xdi[id])

                for (idx, _) in enumerate(connections)                    
                    var.biased_pd[id,idx] = pd_bus[idx]
                    var.biased_qd[id,idx] = qd_bus[idx]
                end

                for (idx, _) in enumerate(connections)                    
                    if abs(pd0[idx]+im*qd0[idx]) == 0.0
                        JuMP.@constraint(model, Xdr[id][:,idx] .== 0)
                        JuMP.@constraint(model, Xdi[id][:,idx] .== 0)
                    end
                end 

                ## 3 different load types
                if load["model"]==POWER
                    for (idx, _) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==pd0[idx])
                        JuMP.@constraint(model, qd[idx]==qd0[idx])
                    end
                elseif load["model"]==IMPEDANCE                    
                    for (idx,_) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==3*a[idx]*var.biased_w[i,idx])
                        JuMP.@constraint(model, qd[idx]==3*b[idx]*var.biased_w[i,idx])
                    end
                elseif load["model"]==CURRENT 
                    for (idx,_) in enumerate(connections)
                        JuMP.@constraint(model, pd[idx]==sqrt(3)/2*a[idx]*(var.biased_w[i,idx]+1))
                        JuMP.@constraint(model, qd[idx]==sqrt(3)/2*b[idx]*(var.biased_w[i,idx]+1))
                    end
                end
 
            end 

            if load["configuration"] == WYE                                
                for (idx,c) in enumerate(connections)  
                    if load["model"]==POWER                                                        
                        var.biased_pd[id,c] = pd0[idx]
                        var.biased_qd[id,c] = qd0[idx]            
                        
                    elseif load["model"]==IMPEDANCE                                                            
                        var.biased_pd[id,c] = a[idx] * var.biased_w[i,c]
                        var.biased_qd[id,c] = b[idx] * var.biased_w[i,c]    
                        
                    elseif load["model"]==CURRENT                         
                        var.biased_pd[id,c] = JuMP.@variable(model, base_name="pd_$(id)[$(c)]")
                        var.biased_qd[id,c] = JuMP.@variable(model, base_name="qd_$(id)[$(c)]")                         
        
                        PMD.set_lower_bound(var.biased_pd[id,c], pmin[idx])
                        PMD.set_upper_bound(var.biased_pd[id,c], pmax[idx])
                        PMD.set_lower_bound(var.biased_qd[id,c], qmin[idx])
                        PMD.set_upper_bound(var.biased_qd[id,c], qmax[idx])
                        
                        JuMP.@constraint(model, var.biased_pd[id,c]==1/2*a[idx]*(var.biased_w[i,c]+1))
                        JuMP.@constraint(model, var.biased_qd[id,c]==1/2*b[idx]*(var.biased_w[i,c]+1))                        
                    end  
                end
            end             
        end
    end     
    
    if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
        Pt=Dict(); Qt=Dict()
        for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)                     
            (Pt, Qt) = PMD.variable_mx_complex(model, [(l,i1,i2)], Dict((l,i1,i2)=>connections) , Dict((l,i1,i2)=>connections); name=("Pt_hat", "Qt_hat"))
            for c in connections
                idx = info.map_line_idx[(l,i1,i2),c]
                var.biased_p_hat[(l,i1,i2),c] = diag(Pt[(l,i1,i2)])[c]
                var.biased_q_hat[(l,i1,i2),c] = diag(Qt[(l,i1,i2)])[c]                      
            end            
        end             
    end
    if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
        for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)  
            for c in connections
                idx = info.map_line_idx[(l,i1,i2),c]
                var.biased_p_hat[(l,i1,i2),c] = JuMP.@variable(model, base_name="p_hat_$((l,i1,i2))[$(c)]")
                var.biased_q_hat[(l,i1,i2),c] = JuMP.@variable(model, base_name="q_hat_$((l,i1,i2))[$(c)]")
            end
        end
    end
      
    bus_arcs = PMD.ref(pm, nw, :bus_arcs_conns_branch, i)
    bus_arcs_sw = PMD.ref(pm, nw, :bus_arcs_conns_switch, i)
    bus_arcs_trans = PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)
    bus_gens = PMD.ref(pm, nw, :bus_conns_gen, i)
    bus_storage = PMD.ref(pm, nw, :bus_conns_storage, i)
    bus_loads = PMD.ref(pm, nw, :bus_conns_load, i)
    bus_shunts = PMD.ref(pm, nw, :bus_conns_shunt, i)
    
    ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]
    
    for (idx,t) in ungrounded_terminals    
        
        JuMP.@constraint(model,
          sum( var.biased_p_hat[a,t] for (a, conns) in bus_arcs if t in conns)        
        + sum( var.biased_p_hat[a,t] for (a, conns) in bus_arcs_trans if t in conns)
        - sum( var.biased_pg_hat[g,t] for (g, conns) in bus_gens if t in conns)        
        + sum( var.biased_pd[d,t] for (d, conns) in bus_loads if t in conns)
        + sum(diag(PMD.ref(pm, nw, :shunt, sh, "gs"))[findfirst(isequal(t), conns)]*var.biased_w[i,t] for (sh, conns) in bus_shunts if t in conns)
        == 0.0, base_name="pb1[$(t)]" )
        JuMP.@constraint(model,
              sum( var.biased_q_hat[a,t] for (a, conns) in bus_arcs if t in conns)            
            + sum( var.biased_q_hat[a,t] for (a, conns) in bus_arcs_trans if t in conns)
            - sum( var.biased_qg_hat[g,t] for (g, conns) in bus_gens if t in conns)            
            + sum( var.biased_qd[d,t] for (d, conns) in bus_loads if t in conns)
            - sum(diag(PMD.ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(t), conns)]*var.biased_w[i,t] for (sh, conns) in bus_shunts if t in conns)
            == 0.0, base_name="pb2[$(t)]")
    end
end


########################################################
#### component-based decomposed LP models
########################################################
function construct_consensus_LP_model(pm, info; nw=0)

    var = Variables()        
      
    for (l,transformer) in PMD.ref(pm, nw, :transformer)                      
        construct_trans_LP_model(pm, info, transformer, l, pm.model, var)        
    end
    for (l,branch) in PMD.ref(pm, nw, :branch)        
        construct_branch_LP_model(pm, info, branch, l,  pm.model, var)        
    end
    for (i,bus) in PMD.ref(pm, nw, :bus)              
        construct_bus_model(pm, info, bus, i, pm.model, var)                
    end  
    for (i,gen) in PMD.ref(pm, nw, :gen)                
        construct_gen_model(pm, info, gen, i, pm.model, var)            
    end       

    ## Consensus
    for (i,gen) in PMD.ref(pm, nw, :gen)                
        for c in gen["connections"]            
            JuMP.@constraint(pm.model, var.pg[i,c] == var.pg_hat[i,c], base_name="CS_pg_[$(i),$(c)]" )
            JuMP.@constraint(pm.model, var.qg[i,c] == var.qg_hat[i,c], base_name="CS_qg_[$(i),$(c)]" )
        end
    end    

    for (l,transformer) in PMD.ref(pm, nw, :transformer)  
        i1 = transformer["f_bus"]
        i2 = transformer["t_bus"]
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)

        for c in transformer["f_connections"]
            JuMP.@constraint(pm.model, var.p[f_idx,c] == var.p_hat[f_idx,c], base_name="CS_p_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[f_idx,c] == var.q_hat[f_idx,c], base_name="CS_q_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[f_idx,c] == var.w[i1,c], base_name="CS_w_[$(f_idx),$(c)]"  )

            JuMP.@constraint(pm.model, var.p[t_idx,c] == var.p_hat[t_idx,c], base_name="CS_p_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[t_idx,c] == var.q_hat[t_idx,c], base_name="CS_q_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[t_idx,c] == var.w[i2,c], base_name="CS_w_[$(t_idx),$(c)]"  )
        end
    end

    for (l,branch) in PMD.ref(pm, nw, :branch)  
        i1 = branch["f_bus"]
        i2 = branch["t_bus"]
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)

        for c in branch["f_connections"]
            JuMP.@constraint(pm.model, var.p[f_idx,c] == var.p_hat[f_idx,c], base_name="CS_p_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[f_idx,c] == var.q_hat[f_idx,c], base_name="CS_q_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[f_idx,c] == var.w[i1,c], base_name="CS_w_[$(f_idx),$(c)]"  )

            JuMP.@constraint(pm.model, var.p[t_idx,c] == var.p_hat[t_idx,c], base_name="CS_p_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[t_idx,c] == var.q_hat[t_idx,c], base_name="CS_q_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[t_idx,c] == var.w[i2,c], base_name="CS_w_[$(t_idx),$(c)]"  )
        end
    end
     
    return pm, var
end
 
function construct_initial_consensus_LP_model(pm, info; nw=0)

    var = Variables()        
      
    for (l,transformer) in PMD.ref(pm, nw, :transformer)                      
        construct_trans_LP_model(pm, info, transformer, l, pm.model, var)        
    end
    for (l,branch) in PMD.ref(pm, nw, :branch)        
        construct_branch_LP_model(pm, info, branch, l,  pm.model, var)        
    end
    for (i,bus) in PMD.ref(pm, nw, :bus)              
        construct_initial_bus_model(pm, info, bus, i, pm.model, var)                
    end  
    for (i,gen) in PMD.ref(pm, nw, :gen)                
        construct_gen_model(pm, info, gen, i, pm.model, var)            
    end       

    ## Consensus
    for (i,gen) in PMD.ref(pm, nw, :gen)                
        for c in gen["connections"]            
            JuMP.@constraint(pm.model, var.pg[i,c] == var.pg_hat[i,c], base_name="CS_pg_[$(i),$(c)]" )
            JuMP.@constraint(pm.model, var.qg[i,c] == var.qg_hat[i,c], base_name="CS_qg_[$(i),$(c)]" )
        end
    end    

    for (l,transformer) in PMD.ref(pm, nw, :transformer)  
        i1 = transformer["f_bus"]
        i2 = transformer["t_bus"]
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)

        for c in transformer["f_connections"]
            JuMP.@constraint(pm.model, var.p[f_idx,c] == var.p_hat[f_idx,c], base_name="CS_p_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[f_idx,c] == var.q_hat[f_idx,c], base_name="CS_q_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[f_idx,c] == var.w[i1,c], base_name="CS_w_[$(f_idx),$(c)]"  )

            JuMP.@constraint(pm.model, var.p[t_idx,c] == var.p_hat[t_idx,c], base_name="CS_p_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[t_idx,c] == var.q_hat[t_idx,c], base_name="CS_q_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[t_idx,c] == var.w[i2,c], base_name="CS_w_[$(t_idx),$(c)]"  )
        end
    end

    for (l,branch) in PMD.ref(pm, nw, :branch)  
        i1 = branch["f_bus"]
        i2 = branch["t_bus"]
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)

        for c in branch["f_connections"]
            JuMP.@constraint(pm.model, var.p[f_idx,c] == var.p_hat[f_idx,c], base_name="CS_p_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[f_idx,c] == var.q_hat[f_idx,c], base_name="CS_q_[$(f_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[f_idx,c] == var.w[i1,c], base_name="CS_w_[$(f_idx),$(c)]"  )

            JuMP.@constraint(pm.model, var.p[t_idx,c] == var.p_hat[t_idx,c], base_name="CS_p_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.q[t_idx,c] == var.q_hat[t_idx,c], base_name="CS_q_[$(t_idx),$(c)]"  )
            JuMP.@constraint(pm.model, var.w_hat[t_idx,c] == var.w[i2,c], base_name="CS_w_[$(t_idx),$(c)]"  )
        end
    end
     
    return pm, var
end

function construct_line_LP_model(pm, info, var; nw=0)
           
    optimizer = JuMP.optimizer_with_attributes(CPLEX.Optimizer)       
    
    Line_info=Dict()
    tmpcnt = 1
    trans_model=Dict()    
    for (l,transformer) in PMD.ref(pm, nw, :transformer)
        trans_model[l] = JuMP.Model()   
        JuMP.set_optimizer(trans_model[l], optimizer)        
        JuMP.set_silent(trans_model[l])              
        construct_trans_LP_model(pm, info, transformer, l, trans_model[l], var)                
        Line_info[tmpcnt] = ["transformer", l, trans_model[l], transformer["f_bus"], transformer["t_bus"], transformer["f_connections"]]
        tmpcnt += 1
    end

    branch_model=Dict()
    for (l,branch) in PMD.ref(pm, nw, :branch)
        branch_model[l] = JuMP.Model()      
        JuMP.set_optimizer(branch_model[l], optimizer)        
        JuMP.set_silent(branch_model[l])                         
        construct_branch_LP_model(pm, info, branch, l,  branch_model[l], var)                
        Line_info[tmpcnt] = ["branch", l, branch_model[l], branch["f_bus"], branch["t_bus"], branch["f_connections"]]
        tmpcnt += 1
    end 
     
    return Line_info
end 

function construct_bus_model(pm, info, var; nw=0)
    optimizer = JuMP.optimizer_with_attributes(CPLEX.Optimizer)    
    bus_model=Dict()
    biased_bus_model=Dict()
    for (i,bus) in PMD.ref(pm, nw, :bus)
        bus_model[i]= JuMP.Model()                 
        JuMP.set_optimizer(bus_model[i], optimizer)        
        JuMP.set_silent(bus_model[i])     
        construct_bus_model(pm, info, bus, i, bus_model[i], var)        
        
        biased_bus_model[i] = JuMP.Model()                         
        JuMP.set_optimizer(biased_bus_model[i], optimizer)               
        JuMP.set_silent(biased_bus_model[i])             
        construct_biased_bus_model(pm, info, bus, i, biased_bus_model[i], var)        
    end    

    return bus_model, biased_bus_model
end


 

