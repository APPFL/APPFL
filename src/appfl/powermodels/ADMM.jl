using Random, Printf, Statistics, Distributions, SCS
include("./Functions.jl")

function save_previous_solution_primal(param)
    ## First-block (Generator + Line)
    param.pg_prev       .= param.pg
    param.qg_prev       .= param.qg
    param.p_prev        .= param.p
    param.q_prev        .= param.q
    param.w_hat_prev    .= param.w_hat
    ## Second-block (Bus)
    param.pg_hat_prev   .= param.pg_hat
    param.qg_hat_prev   .= param.qg_hat
    param.p_hat_prev    .= param.p_hat
    param.q_hat_prev    .= param.q_hat
    param.w_prev        .= param.w     
end

function save_previous_solution_dual(param)     
    param.λ_pg_prev .= param.λ_pg
    param.λ_qg_prev .= param.λ_qg
    param.λ_p_prev .= param.λ_p
    param.λ_q_prev .= param.λ_q
    param.λ_w_prev .= param.λ_w
end

function solve_generator_closed_form(pm, info, param;nw=0)
    for (i,gen) in PMD.ref(pm, nw, :gen)        
        for c in gen["connections"]
            idx = info.map_gen_idx[i,c]
            param.pg[idx] = param.pg_hat[idx] - (1.0/param.ρ_pg[idx])*(1.0 + param.λ_pg[idx])
            param.qg[idx] = param.qg_hat[idx] - (1.0/param.ρ_qg[idx])*param.λ_qg[idx]

            param.pg[idx] = clamp( param.pg[idx], gen["pmin"][c], gen["pmax"][c] )
            param.qg[idx] = clamp( param.qg[idx], gen["qmin"][c], gen["qmax"][c] )
        end
    end     
end

function solve_line(pm, info, param, Line_info, var; nw=0)
 
    for line_idx = 1:info.NLines

        model       = Line_info[line_idx][3]
        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]  

        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)
        org_obj_fn = JuMP.objective_function(model)            
        
        additional_term = 0
        for c in connections
            line_f_idx = info.map_line_idx[f_idx,c]
            line_t_idx = info.map_line_idx[t_idx,c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]

            term_1 = param.λ_p[line_f_idx]*var.p[f_idx,c] + param.λ_p[line_t_idx]*var.p[t_idx,c]            
            term_2 = param.λ_q[line_f_idx]*var.q[f_idx,c] + param.λ_q[line_t_idx]*var.q[t_idx,c]            
            term_3 = param.λ_w[line_f_idx]*var.w_hat[f_idx,c] + param.λ_w[line_t_idx]*var.w_hat[t_idx,c]

            term_1_rho = 0.5*param.ρ_p[line_f_idx]*( var.p[f_idx,c] - param.p_hat[line_f_idx] )^2 + 0.5*param.ρ_p[line_t_idx]*( var.p[t_idx,c] - param.p_hat[line_t_idx] )^2
            term_2_rho = 0.5*param.ρ_q[line_f_idx]*( var.q[f_idx,c] - param.q_hat[line_f_idx] )^2 + 0.5*param.ρ_q[line_t_idx]*( var.q[t_idx,c] - param.q_hat[line_t_idx] )^2
            term_3_rho = 0.5*param.ρ_w[line_f_idx]*( var.w_hat[f_idx,c] - param.w[bus_f_idx] )^2  + 0.5*param.ρ_w[line_t_idx]*( var.w_hat[t_idx,c] - param.w[bus_t_idx] )^2

            additional_term += term_1 + term_2 + term_3 + term_1_rho + term_2_rho + term_3_rho                
        end            
                
        JuMP.set_objective_function(model, org_obj_fn + additional_term)                        
        JuMP.optimize!(model)
        if JuMP.termination_status(model) != OPTIMAL
            println("Status (trans)=", JuMP.termination_status(model))
            println(model)            
        end 
        
        for c in connections          
            line_f_idx = info.map_line_idx[f_idx,c]
            line_t_idx = info.map_line_idx[t_idx,c]                

            param.p[line_f_idx] = JuMP.value( var.p[f_idx,c]  )
            param.p[line_t_idx] = JuMP.value( var.p[t_idx,c]  )                
            param.q[line_f_idx] = JuMP.value( var.q[f_idx,c]  )
            param.q[line_t_idx] = JuMP.value( var.q[t_idx,c]  )
            param.w_hat[line_f_idx] = JuMP.value( var.w_hat[f_idx,c] )
            param.w_hat[line_t_idx] = JuMP.value( var.w_hat[t_idx,c] )
        end

        JuMP.set_objective_function(model, org_obj_fn)       
    end 

end

function solve_bus(pm, info, param, bus_model, var; nw=0)
    for (i,bus) in PMD.ref(pm, nw, :bus)                
 
        model = bus_model[i]          

        org_obj_fn = JuMP.objective_function(model)   
   
        additional_term = 0
        for c in bus["terminals"]
            
            idx = info.map_bus_idx[i,c]
            additional_term += (0.5/info.eta) * (var.w[i,c] - param.w_prev[idx])^2
            
            ## Generator
            if i in keys(info.map_bus_to_gen)
                gen_id = info.map_bus_to_gen[i]
                gen = PMD.ref(pm, nw, :gen, gen_id)  
                if c in gen["connections"]      
                    idx = info.map_gen_idx[gen_id,c]

                    term_1 = - param.λ_pg[idx]*var.pg_hat[gen_id,c] - param.λ_qg[idx]*var.qg_hat[gen_id,c]
                    term_2 = 0.5*param.ρ_pg[idx]*( param.pg[idx] - var.pg_hat[gen_id,c] )^2 + 0.5*param.ρ_qg[idx]*( param.qg[idx] - var.qg_hat[gen_id,c] )^2
                    term_3 =  (0.5/info.eta) * ( (var.pg_hat[gen_id,c] - param.pg_hat_prev[idx])^2 + (var.qg_hat[gen_id,c] - param.qg_hat_prev[idx])^2 )   
                    additional_term += term_1 + term_2 + term_3
                end                                                  
            end
            
            ## Transformer
            if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)       
                    if c in connections    
                        idx = info.map_line_idx[(l,i1,i2),c]
                        
                        term_1 = - param.λ_p[idx]*var.p_hat[(l,i1,i2),c] - param.λ_q[idx]*var.q_hat[(l,i1,i2),c]  - param.λ_w[idx]*var.w[i,c]
                        term_2 = 0.5*param.ρ_p[idx]*(param.p[idx] - var.p_hat[(l,i1,i2),c])^2+ 0.5*param.ρ_q[idx]*(param.q[idx] - var.q_hat[(l,i1,i2),c])^2 + 0.5*param.ρ_w[idx]*(param.w_hat[idx] - var.w[i,c])^2
                        term_3 = (0.5/info.eta) * ( (var.p_hat[(l,i1,i2),c] - param.p_hat_prev[idx])^2 
                        + (var.q_hat[(l,i1,i2),c] - param.q_hat_prev[idx])^2 )
                        additional_term += term_1  + term_2 + term_3
                    end 
                end             
            end
                        
            ## Branch
            if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)                      
                    if c in connections                                                 
                        idx = info.map_line_idx[(l,i1,i2),c]

                        term_1 = - param.λ_p[idx]*var.p_hat[(l,i1,i2),c] - param.λ_q[idx]*var.q_hat[(l,i1,i2),c] - param.λ_w[idx]*var.w[i,c] 
                        term_2 = 0.5*param.ρ_p[idx]*(param.p[idx] - var.p_hat[(l,i1,i2),c])^2+ 0.5*param.ρ_q[idx]*(param.q[idx] - var.q_hat[(l,i1,i2),c])^2 + 0.5*param.ρ_w[idx]*(param.w_hat[idx] - var.w[i,c])^2
                        term_3 = (0.5/info.eta) * ( (var.p_hat[(l,i1,i2),c] - param.p_hat_prev[idx])^2 + (var.q_hat[(l,i1,i2),c] - param.q_hat_prev[idx])^2 )
                        additional_term += term_1 + term_2 + term_3
                         
                    end                                                                     
                end             
            end
        end  
         
        JuMP.set_objective_function(model, org_obj_fn + additional_term)                
        JuMP.optimize!(model)
        if JuMP.termination_status(model) != OPTIMAL
            println("Status (bus)=", JuMP.termination_status(model))
        end
        
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
        JuMP.set_objective_function(model, org_obj_fn)      
    end    
end 

function solve_bus_closed_form(pm, info, param, var, β; nw=0)
    w = ones(info.nbus)
    pg_hat = zeros(info.ngen)
    qg_hat = zeros(info.ngen)
    p_hat = zeros(2*(info.nbranch+info.ntrans))
    q_hat = zeros(2*(info.nbranch+info.ntrans))
    
    for i = 1:info.NBuses                                    
        bus = PMD.ref(pm, nw, :bus, i)  

        if i in info.Delta_bus  ## Case 1: DELTA Configuration
            gs = zeros(3); bs = zeros(3)                     
            for c in bus["terminals"]
                for (sh, conns) in PMD.ref(pm, nw, :bus_conns_shunt, i)                            
                    if c in conns
                        gs[c] = diag(PMD.ref(pm, nw, :shunt, sh, "gs"))[findfirst(isequal(c), conns)]                                
                        bs[c] = diag(PMD.ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(c), conns)]                                 
                    end
                end 
            end 
              
            ## Case 1-1: Delta bus with one positive load
            if i in info.Special_Delta_bus
                id = info.map_bus_to_load[i][1]
                load = PMD.ref(pm, nw, :load, id)                            
    
                ## Perturbed                                    
                pd0 = load["pd"].*β
                qd0 = load["qd"]   
    
                a, alpha, b, beta = load_expmodel_params(load, bus, pd0, qd0)
                        
                vmin, vmax = PMD._calc_load_vbounds(load, bus)            
                pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
                connections = load["connections"]  

                @assert pd0[connections[1]] > 0.0
                @assert pd0[connections[2]] == 0.0
                @assert pd0[connections[3]] == 0.0

                CP = zeros(3); GP = zeros(3); CQ = zeros(3); GQ = zeros(3); 
                 
                for (idx,c) in enumerate(connections)                                 
                    if load["model"]==POWER     
                        CP[idx] = pd0[idx]; GP[idx] = 0.0;
                        CQ[idx] = qd0[idx]; GQ[idx] = 0.0;      
                    elseif load["model"]==IMPEDANCE 
                        CP[idx] = 0.0; GP[idx] = 3.0*a[idx] ;
                        CQ[idx] = 0.0; GQ[idx] = 3.0*b[idx] ;    
                    elseif load["model"]==CURRENT                         
                        CP[idx] = sqrt(3)/2*a[idx]; GP[idx] = sqrt(3)/2*a[idx] ;
                        CQ[idx] = sqrt(3)/2*b[idx]; GQ[idx] = sqrt(3)/2*b[idx] ;    
                    end
                end
                 
                w, pg_hat, qg_hat, p_hat, q_hat = calculate_solution_special_delta(info, param, i, connections, gs, bs, CP, GP, CQ, GQ, w, pg_hat, qg_hat, p_hat, q_hat)               
  
            ## Case 1-2: Rest    
            else
                CP = zeros(3); GP = zeros(3); CQ = zeros(3); GQ = zeros(3); 
                for id ∈ info.map_bus_to_load[i] 
                    load = PMD.ref(pm, nw, :load, id)                            
            
                    ## Perturbed                                    
                    pd0 = load["pd"].*β
                    qd0 = load["qd"]   
        
                    a, alpha, b, beta = load_expmodel_params(load, bus, pd0, qd0)
            
                    vmin, vmax = PMD._calc_load_vbounds(load, bus)            
                    pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
                    connections = load["connections"]  
        
                    for (idx,c) in enumerate(connections)                                 
                        if load["model"]==POWER     
                            CP[idx] += pd0[idx]; GP[idx] += 0.0;
                            CQ[idx] += qd0[idx]; GQ[idx] += 0.0;      
                        elseif load["model"]==IMPEDANCE 
                            CP[idx] += 0.0; GP[idx] += 3.0*a[idx] ;
                            CQ[idx] += 0.0; GQ[idx] += 3.0*b[idx] ;    
                        elseif load["model"]==CURRENT                         
                            CP[idx] += sqrt(3)/2*a[idx]; GP[idx] += sqrt(3)/2*a[idx] ;
                            CQ[idx] += sqrt(3)/2*b[idx]; GQ[idx] += sqrt(3)/2*b[idx] ;    
                        end
                    end
                end
                w, pg_hat, qg_hat, p_hat, q_hat = calculate_solution_delta(info, param, i, bus["terminals"], gs, bs, CP, GP, CQ, GQ, w, pg_hat, qg_hat, p_hat, q_hat)   
            end
            
        elseif i in info.Wye_bus  ## Case 2: WYE Configuration or No Load for partial phases
            ## Case 2-1: WYE Configuration  
            for id ∈ info.map_bus_to_load[i]                 
                load = PMD.ref(pm, nw, :load, id)                            

                ## Perturbed                                    
                pd0 = load["pd"].*β
                qd0 = load["qd"]   
    
                a, alpha, b, beta = load_expmodel_params(load, bus, pd0, qd0)
        
                vmin, vmax = PMD._calc_load_vbounds(load, bus)            
                pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
                connections = load["connections"]                  
                                        
                CP = 0.0; GP = 0.0; CQ = 0.0; GQ = 0.0; 
                for (idx,c) in enumerate(connections)  
                    gs=0.0; bs=0.0;
                    for (sh, conns) in PMD.ref(pm, nw, :bus_conns_shunt, i)                            
                        if c in conns
                            gs = diag(PMD.ref(pm, nw, :shunt, sh, "gs"))[findfirst(isequal(c), conns)]                                
                            bs = diag(PMD.ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(c), conns)] 
                        end
                    end 

                    if load["model"]==POWER     
                        CP = pd0[idx]; GP = 0.0;
                        CQ = qd0[idx]; GQ = 0.0;      
                    elseif load["model"]==IMPEDANCE 
                        CP = 0.0; GP = a[idx] ;
                        CQ = 0.0; GQ = b[idx] ;    
                    elseif load["model"]==CURRENT                         
                        CP = 0.5*a[idx]; GP = 0.5*a[idx] ;
                        CQ = 0.5*b[idx]; GQ = 0.5*b[idx] ;    
                    end
                    w, pg_hat, qg_hat, p_hat, q_hat = calculate_solution_wye(info, param, i, c, gs, bs, CP, GP, CQ, GQ, w, pg_hat, qg_hat, p_hat, q_hat)    
                end                                   
            end 

            ## Case 2-2: No Load for partial phases
            for c in bus["terminals"]   
                tmpcnt = 0
                for id ∈ info.map_bus_to_load[i]   
                    load = PMD.ref(pm, nw, :load, id)                             
                    for t in load["connections"]                          
                        if c == t
                            tmpcnt = 1
                        end
                    end
                end                
                if tmpcnt == 0                    
                    gs=0.0; bs=0.0;
                    for (sh, conns) in PMD.ref(pm, nw, :bus_conns_shunt, i)                            
                        if c in conns
                            gs = diag(PMD.ref(pm, nw, :shunt, sh, "gs"))[findfirst(isequal(c), conns)]                                
                            bs = diag(PMD.ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(c), conns)] 
                        end
                    end                     
                    w, pg_hat, qg_hat, p_hat, q_hat = calculate_solution_wye(info, param, i, c, gs, bs, 0.0, 0.0, 0.0, 0.0, w, pg_hat, qg_hat, p_hat, q_hat)  
                end
            end

        else  ## Case 3: No Load for all phases
            for c in bus["terminals"]      
                gs=0.0; bs=0.0;
                for (sh, conns) in PMD.ref(pm, nw, :bus_conns_shunt, i)                            
                    if c in conns
                        gs = diag(PMD.ref(pm, nw, :shunt, sh, "gs"))[findfirst(isequal(c), conns)]                                
                        bs = diag(PMD.ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(c), conns)] 
                    end
                end                           
                w, pg_hat, qg_hat, p_hat, q_hat = calculate_solution_wye(info, param, i, c, gs, bs, 0.0, 0.0, 0.0, 0.0, w, pg_hat, qg_hat, p_hat, q_hat) 
            end     
        end 

    end
    return w, pg_hat, qg_hat, p_hat, q_hat
end 
 
function calculate_solution_wye(info, param, i, c, gs, bs, CP, GP, CQ, GQ, w, pg_hat, qg_hat, p_hat, q_hat)
    
    bus_idx = info.map_bus_idx[i,c]
    ## Calculate dual solution
    ρ_p = 0.0; ρ_q = 0.0; ρ_w = 0.0; 
    d_p = 0.0; d_w = 0.0; d_q = 0.0;    
    for line_idx in info.setA[i,c]
        idx = info.map_line_idx[line_idx,c]        
        ρ_p += 1.0 / (param.ρ_p[idx] + 1.0/info.eta)
        ρ_q += 1.0 / (param.ρ_q[idx] + 1.0/info.eta)
        ρ_w += param.ρ_w[idx]
        d_p += 1.0/(param.ρ_p[idx] + 1.0/info.eta) * ( param.λ_p[idx] + param.ρ_p[idx]*param.p[idx] + param.p_hat_prev[idx]/info.eta  )
        d_q += 1.0/(param.ρ_q[idx] + 1.0/info.eta) * ( param.λ_q[idx] + param.ρ_q[idx]*param.q[idx] + param.q_hat_prev[idx]/info.eta  )
        d_w += param.λ_w[idx] + param.ρ_w[idx]*param.w_hat[idx]
    end
    ρ_w += 1.0/info.eta
    d_w += param.w_prev[bus_idx]/info.eta
    
    ρ_pg = 0.0; ρ_qg = 0.0;
    d_pg = 0.0; d_qg = 0.0;
    for gen_idx in info.setG[i,c]
        idx = info.map_gen_idx[gen_idx,c]
        ρ_pg += 1.0 / (param.ρ_pg[idx] + 1.0/info.eta)
        ρ_qg += 1.0 / (param.ρ_qg[idx] + 1.0/info.eta)
        
        d_pg += 1.0/(param.ρ_pg[idx] + 1.0/info.eta) * ( param.λ_pg[idx] + param.ρ_pg[idx]*param.pg[idx] + param.pg_hat_prev[idx]/info.eta)
        d_qg += 1.0/(param.ρ_qg[idx] + 1.0/info.eta) * ( param.λ_qg[idx] + param.ρ_qg[idx]*param.qg[idx] + param.qg_hat_prev[idx]/info.eta)
    end
    
    D11 = ρ_p + (GP+gs)*(GP+gs)/ρ_w + ρ_pg
    D12 = (GP+gs)*(GQ-bs)/ρ_w
    D21 = (GQ-bs)*(GP+gs)/ρ_w
    D22 = ρ_q + (GQ-bs)*(GQ-bs)/ρ_w + ρ_qg
    d1 = d_p + CP + ((GP+gs)/ρ_w)*d_w - d_pg 
    d2 = d_q + CQ + ((GQ-bs)/ρ_w)*d_w - d_qg
     
    μ_p =  ( D22*d1 - D12*d2) / (D11*D22 - D12*D21)
    μ_q =  (-D21*d1 + D11*d2) / (D11*D22 - D12*D21)
    
    ## Calculate primal solution
    w[bus_idx] = (1.0/ρ_w)*( d_w - (GP+gs)*μ_p - (GQ-bs)*μ_q )         
     
    for line_idx in info.setA[i,c]
        idx = info.map_line_idx[line_idx,c]                        
        p_hat[idx] = 1.0 / (param.ρ_p[idx]+1.0/info.eta) * ( param.λ_p[idx] + param.ρ_p[idx]*param.p[idx] + param.p_hat_prev[idx]/info.eta - μ_p )
        q_hat[idx] = 1.0 / (param.ρ_q[idx]+1.0/info.eta) * ( param.λ_q[idx] + param.ρ_q[idx]*param.q[idx] + param.q_hat_prev[idx]/info.eta - μ_q )        
    end
    for gen_idx in info.setG[i,c]
        idx = info.map_gen_idx[gen_idx,c]
        pg_hat[idx] = 1.0 / (param.ρ_pg[idx]+1.0/info.eta) * ( param.λ_pg[idx] + param.ρ_pg[idx]*param.pg[idx] + param.pg_hat_prev[idx]/info.eta + μ_p )
        qg_hat[idx] = 1.0 / (param.ρ_qg[idx]+1.0/info.eta) * ( param.λ_qg[idx] + param.ρ_qg[idx]*param.qg[idx] + param.qg_hat_prev[idx]/info.eta + μ_q )        
    end
    
    return w, pg_hat, qg_hat, p_hat, q_hat
end
 
function calculate_solution_delta(info, param, i, connections, gs, bs, CP, GP, CQ, GQ, w, pg_hat, qg_hat, p_hat, q_hat)       
    
    ## Calculate dual solution
    D11 = 0.0; D12 = 0.0; D21 = 0.0; D22 = 0.0; d1=0.0; d2=0.0;
    for (c,_) in enumerate(connections)   
        # println( "i=", i, " c=", c)   
    
        if info.setA[i,c] != []
            bus_idx = info.map_bus_idx[i,c]

            ρ_p = 0.0; ρ_q = 0.0; ρ_w = 0.0; 
            d_p = 0.0; d_w = 0.0; d_q = 0.0;    
            for line_idx in info.setA[i,c]
                idx = info.map_line_idx[line_idx,c]        
                 
                ρ_p += 1.0 / (param.ρ_p[idx] + 1.0/info.eta)
                ρ_q += 1.0 / (param.ρ_q[idx] + 1.0/info.eta)
                ρ_w += param.ρ_w[idx]
                d_p += 1.0/(param.ρ_p[idx] + 1.0/info.eta) * ( param.λ_p[idx] + param.ρ_p[idx]*param.p[idx] + param.p_hat_prev[idx]/info.eta  )
                d_q += 1.0/(param.ρ_q[idx] + 1.0/info.eta) * ( param.λ_q[idx] + param.ρ_q[idx]*param.q[idx] + param.q_hat_prev[idx]/info.eta  )
                d_w += param.λ_w[idx] + param.ρ_w[idx]*param.w_hat[idx]
            end
            ρ_w += 1.0/info.eta
            d_w += param.w_prev[bus_idx]/info.eta

            ρ_pg = 0.0; ρ_qg = 0.0;
            d_pg = 0.0; d_qg = 0.0;
            for gen_idx in info.setG[i,c]
                idx = info.map_gen_idx[gen_idx,c]
                
                ρ_pg += 1.0 / (param.ρ_pg[idx] + 1.0/info.eta)
                ρ_qg += 1.0 / (param.ρ_qg[idx] + 1.0/info.eta)
                
                d_pg += 1.0/(param.ρ_pg[idx] + 1.0/info.eta) * ( param.λ_pg[idx] + param.ρ_pg[idx]*param.pg[idx] + param.pg_hat_prev[idx]/info.eta)
                d_qg += 1.0/(param.ρ_qg[idx] + 1.0/info.eta) * ( param.λ_qg[idx] + param.ρ_qg[idx]*param.qg[idx] + param.qg_hat_prev[idx]/info.eta)
            end
 
            D11 += ρ_p + (GP[c] + gs[c])*(GP[c] + gs[c])/ρ_w + ρ_pg            
            D12 += (GP[c] + gs[c])*(GQ[c] - bs[c])/ρ_w
            D21 += (GQ[c] - bs[c])*(GP[c] + gs[c])/ρ_w
            D22 += ρ_q + (GQ[c] - bs[c])*(GQ[c] - bs[c])/ρ_w + ρ_qg
            
            d1  += d_p + CP[c] + d_w*(GP[c] + gs[c])/ρ_w - d_pg 
            d2  += d_q + CQ[c] + d_w*(GQ[c] - bs[c])/ρ_w - d_qg            
         
        end          
    end
     
    μ_p =  ( D22*d1 - D12*d2) / (D11*D22 - D12*D21)
    μ_q =  (-D21*d1 + D11*d2) / (D11*D22 - D12*D21)
        
    ## Calculate primal solution
    for (c,_) in enumerate(connections)    
        if info.setA[i,c] != []
            bus_idx = info.map_bus_idx[i,c]

            ρ_w = 0.0; d_w = 0.0;
            for line_idx in info.setA[i,c]
                idx = info.map_line_idx[line_idx,c]                    
                ρ_w += param.ρ_w[idx]
                d_w += param.λ_w[idx] + param.ρ_w[idx]*param.w_hat[idx]
            end
            ρ_w += 1.0/info.eta
            d_w += param.w_prev[bus_idx]/info.eta
            
            w[bus_idx] = (1.0/ρ_w)*( d_w - (GP[c]+gs[c])*μ_p - (GQ[c]-bs[c])*μ_q ) 
             
            for line_idx in info.setA[i,c]
                idx = info.map_line_idx[line_idx,c]                        
                p_hat[idx] = 1.0 / (param.ρ_p[idx]+1.0/info.eta) * ( param.λ_p[idx] + param.ρ_p[idx]*param.p[idx] + param.p_hat_prev[idx]/info.eta - μ_p )
                q_hat[idx] = 1.0 / (param.ρ_q[idx]+1.0/info.eta) * ( param.λ_q[idx] + param.ρ_q[idx]*param.q[idx] + param.q_hat_prev[idx]/info.eta - μ_q )       
            end
            for gen_idx in info.setG[i,c]
                idx = info.map_gen_idx[gen_idx,c]
                pg_hat[idx] = 1.0 / (param.ρ_pg[idx]+1.0/info.eta) * ( param.λ_pg[idx] + param.ρ_pg[idx]*param.pg[idx] + param.pg_hat_prev[idx]/info.eta + μ_p )
                qg_hat[idx] = 1.0 / (param.ρ_qg[idx]+1.0/info.eta) * ( param.λ_qg[idx] + param.ρ_qg[idx]*param.qg[idx] + param.qg_hat_prev[idx]/info.eta + μ_q )
            end            
        end         
    end 
    return w, pg_hat, qg_hat, p_hat, q_hat
end

function calculate_solution_special_delta(info, param, i, connections, gs, bs, CP, GP, CQ, GQ, w, pg_hat, qg_hat, p_hat, q_hat)       

    ψ = connections[1]
    ψ_plus = connections[2]
    ψ_minus = connections[3]
     
    ρ_p_vec = zeros(3); ρ_q_vec = zeros(3); ρ_w_vec = zeros(3); 
    d_p_vec = zeros(3); d_q_vec = zeros(3); d_w_vec = zeros(3);

    ρ_pg_vec = zeros(3); ρ_qg_vec = zeros(3);
    d_pg_vec = zeros(3); d_qg_vec = zeros(3);
    
    ## Calculate dual solution
    for (_,c) in enumerate(connections)           
        if info.setA[i,c] != []
            bus_idx = info.map_bus_idx[i,c]

            ρ_p = 0.0; ρ_q = 0.0; ρ_w = 0.0; 
            d_p = 0.0; d_w = 0.0; d_q = 0.0;    
            for line_idx in info.setA[i,c]
                idx = info.map_line_idx[line_idx,c]        
                 
                ρ_p += 1.0 / (param.ρ_p[idx] + 1.0/info.eta)
                ρ_q += 1.0 / (param.ρ_q[idx] + 1.0/info.eta)
                ρ_w += param.ρ_w[idx]
                d_p += 1.0/(param.ρ_p[idx] + 1.0/info.eta) * ( param.λ_p[idx] + param.ρ_p[idx]*param.p[idx] + param.p_hat_prev[idx]/info.eta  )
                d_q += 1.0/(param.ρ_q[idx] + 1.0/info.eta) * ( param.λ_q[idx] + param.ρ_q[idx]*param.q[idx] + param.q_hat_prev[idx]/info.eta  )
                d_w += param.λ_w[idx] + param.ρ_w[idx]*param.w_hat[idx]
            end
            ρ_w += 1.0/info.eta
            d_w += param.w_prev[bus_idx]/info.eta

            ρ_p_vec[c] = ρ_p; ρ_q_vec[c] = ρ_q; ρ_w_vec[c] = ρ_w;            
            d_p_vec[c] = d_p; d_q_vec[c] = d_q; d_w_vec[c] = d_w;
 
            ρ_pg = 0.0; ρ_qg = 0.0;
            d_pg = 0.0; d_qg = 0.0;
            for gen_idx in info.setG[i,c]
                idx = info.map_gen_idx[gen_idx,c]
                
                ρ_pg += 1.0 / (param.ρ_pg[idx] + 1.0/info.eta)
                ρ_qg += 1.0 / (param.ρ_qg[idx] + 1.0/info.eta)
                
                d_pg += 1.0/(param.ρ_pg[idx] + 1.0/info.eta) * ( param.λ_pg[idx] + param.ρ_pg[idx]*param.pg[idx] + param.pg_hat_prev[idx]/info.eta)
                d_qg += 1.0/(param.ρ_qg[idx] + 1.0/info.eta) * ( param.λ_qg[idx] + param.ρ_qg[idx]*param.qg[idx] + param.qg_hat_prev[idx]/info.eta)
            end
            ρ_pg_vec[c] = ρ_pg; ρ_qg_vec[c] = ρ_qg;
            d_pg_vec[c] = d_pg; d_qg_vec[c] = d_qg;          
        end          
    end
     
    ## (1) First Row
    D11_ψ_minus = 0.0; D11_ψ = 0.0; D11_ψ_plus = 0.0;
    D12_ψ_minus = 0.0; D12_ψ = 0.0; D12_ψ_plus = 0.0;
    D13_ψ_minus = 0.0;              D13_ψ_plus = 0.0;
    D14_ψ_minus = 0.0;              D14_ψ_plus = 0.0;
    d1_ψ_minus  = 0.0; d1_ψ  = 0.0; d1_ψ_plus  = 0.0;
    if ρ_w_vec[ψ_minus] > 0.0  
        D11_ψ_minus = ( GP[ψ_minus] ) * ( GP[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D12_ψ_minus = ( GP[ψ_minus] ) * ( GQ[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D13_ψ_minus = ( GP[ψ_minus] ) * ( gs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D14_ψ_minus = ( GP[ψ_minus] ) * ( -bs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        d1_ψ_minus  = ( GP[ψ_minus] ) * ( d_w_vec[ψ_minus] ) / ρ_w_vec[ψ_minus] 
    end
    if ρ_w_vec[ψ] > 0.0  
        D11_ψ = ( GP[ψ] + gs[ψ] ) * ( GP[ψ] + gs[ψ] ) / ρ_w_vec[ψ]
        D12_ψ = ( GP[ψ] + gs[ψ] ) * ( GQ[ψ] - bs[ψ] ) / ρ_w_vec[ψ]
        d1_ψ  = ( GP[ψ] + gs[ψ] ) * ( d_w_vec[ψ] ) / ρ_w_vec[ψ] 
    end
    if ρ_w_vec[ψ_plus] > 0.0  
        D11_ψ_plus = ( gs[ψ_plus] ) * ( gs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D12_ψ_plus = ( gs[ψ_plus] ) * ( -bs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D13_ψ_plus = ( gs[ψ_plus] ) * ( GP[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D14_ψ_plus = ( gs[ψ_plus] ) * ( GQ[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        d1_ψ_plus  = ( gs[ψ_plus] ) * ( d_w_vec[ψ_plus] ) / ρ_w_vec[ψ_plus] 
    end      

    D11 = D11_ψ_minus + D11_ψ + D11_ψ_plus + ρ_p_vec[ψ] + ρ_pg_vec[ψ] + ρ_p_vec[ψ_plus] + ρ_pg_vec[ψ_plus];                                           
    D12 = D12_ψ_minus + D12_ψ + D12_ψ_plus; 
    D13 = D13_ψ_minus         + D13_ψ_plus; 
    D14 = D14_ψ_minus         + D14_ψ_plus; 
    d1  = d1_ψ_minus + d1_ψ + d1_ψ_plus + d_p_vec[ψ] - d_pg_vec[ψ] + CP[ψ_minus] + d_p_vec[ψ_plus] - d_pg_vec[ψ_plus] + CP[ψ];
                       
    ## (2) Second Row
    D21_ψ_minus = 0.0; D21_ψ = 0.0; D21_ψ_plus = 0.0;
    D22_ψ_minus = 0.0; D22_ψ = 0.0; D22_ψ_plus = 0.0;
    D23_ψ_minus = 0.0;              D23_ψ_plus = 0.0;
    D24_ψ_minus = 0.0;              D24_ψ_plus = 0.0;
    d2_ψ_minus  = 0.0; d2_ψ  = 0.0; d2_ψ_plus  = 0.0;
    if ρ_w_vec[ψ_minus] > 0.0  
        D21_ψ_minus = ( GQ[ψ_minus] ) * ( GP[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D22_ψ_minus = ( GQ[ψ_minus] ) * ( GQ[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D23_ψ_minus = ( GQ[ψ_minus] ) * ( gs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D24_ψ_minus = ( GQ[ψ_minus] ) * ( -bs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        d2_ψ_minus  = ( GQ[ψ_minus] ) * ( d_w_vec[ψ_minus] ) / ρ_w_vec[ψ_minus] 
    end
    if ρ_w_vec[ψ] > 0.0  
        D21_ψ = ( GQ[ψ] - bs[ψ] ) * ( GP[ψ] + gs[ψ] ) / ρ_w_vec[ψ]
        D22_ψ = ( GQ[ψ] - bs[ψ] ) * ( GQ[ψ] - bs[ψ] ) / ρ_w_vec[ψ]
        d2_ψ  = ( GQ[ψ] - bs[ψ] ) * ( d_w_vec[ψ] ) / ρ_w_vec[ψ] 
    end
    if ρ_w_vec[ψ_plus] > 0.0  
        D21_ψ_plus = ( -bs[ψ_plus] ) * ( gs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D22_ψ_plus = ( -bs[ψ_plus] ) * ( -bs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D23_ψ_plus = ( -bs[ψ_plus] ) * ( GP[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D24_ψ_plus = ( -bs[ψ_plus] ) * ( GQ[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        d2_ψ_plus  = ( -bs[ψ_plus] ) * ( d_w_vec[ψ_plus] ) / ρ_w_vec[ψ_plus] 
    end   

    D21 = D21_ψ_minus + D21_ψ + D21_ψ_plus;
    D22 = D22_ψ_minus + D22_ψ + D22_ψ_plus + ρ_q_vec[ψ] + ρ_qg_vec[ψ] + ρ_q_vec[ψ_plus] + ρ_qg_vec[ψ_plus];                                           
    D23 = D23_ψ_minus         + D23_ψ_plus; 
    D24 = D24_ψ_minus         + D24_ψ_plus; 
    d2  = d2_ψ_minus  + d2_ψ  + d2_ψ_plus + d_q_vec[ψ]      - d_qg_vec[ψ]      + CQ[ψ_minus] + d_q_vec[ψ_plus] - d_qg_vec[ψ_plus] + CQ[ψ];
                            

    ## (3) Third Row
    D31_ψ_minus = 0.0;              D31_ψ_plus = 0.0;
    D32_ψ_minus = 0.0;              D32_ψ_plus = 0.0;
    D33_ψ_minus = 0.0;              D33_ψ_plus = 0.0;
    D34_ψ_minus = 0.0;              D34_ψ_plus = 0.0;
    d3_ψ_minus  = 0.0;              d3_ψ_plus  = 0.0;
    if ρ_w_vec[ψ_minus] > 0.0  
        D31_ψ_minus = ( gs[ψ_minus] ) * ( GP[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D32_ψ_minus = ( gs[ψ_minus] ) * ( GQ[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D33_ψ_minus = ( gs[ψ_minus] ) * ( gs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D34_ψ_minus = ( gs[ψ_minus] ) * ( -bs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        d3_ψ_minus  = ( gs[ψ_minus] ) * ( d_w_vec[ψ_minus] ) / ρ_w_vec[ψ_minus] 
    end    
    if ρ_w_vec[ψ_plus] > 0.0  
        D31_ψ_plus = ( GP[ψ_plus] ) * ( gs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D32_ψ_plus = ( GP[ψ_plus] ) * ( -bs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D33_ψ_plus = ( GP[ψ_plus] ) * ( GP[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D34_ψ_plus = ( GP[ψ_plus] ) * ( GQ[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        d3_ψ_plus  = ( GP[ψ_plus] ) * ( d_w_vec[ψ_plus] ) / ρ_w_vec[ψ_plus] 
    end      
      
    D31 = D31_ψ_minus + D31_ψ_plus;                                    
    D32 = D32_ψ_minus + D32_ψ_plus; 
    D33 = D33_ψ_minus + D33_ψ_plus + ρ_p_vec[ψ_minus] + ρ_pg_vec[ψ_minus]; 
    D34 = D34_ψ_minus + D34_ψ_plus; 
    d3  = d3_ψ_minus  + d3_ψ_plus  + d_p_vec[ψ_minus] - d_pg_vec[ψ_minus] + CP[ψ_plus];
                             
    ## (4) Fourth Row 
    D41_ψ_minus = 0.0;              D41_ψ_plus = 0.0;
    D42_ψ_minus = 0.0;              D42_ψ_plus = 0.0;
    D43_ψ_minus = 0.0;              D43_ψ_plus = 0.0;
    D44_ψ_minus = 0.0;              D44_ψ_plus = 0.0;
    d4_ψ_minus  = 0.0;              d4_ψ_plus  = 0.0;
    if ρ_w_vec[ψ_minus] > 0.0  
        D41_ψ_minus = ( -bs[ψ_minus] ) * ( GP[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D42_ψ_minus = ( -bs[ψ_minus] ) * ( GQ[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D43_ψ_minus = ( -bs[ψ_minus] ) * ( gs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        D44_ψ_minus = ( -bs[ψ_minus] ) * ( -bs[ψ_minus] ) / ρ_w_vec[ψ_minus] 
        d4_ψ_minus  = ( -bs[ψ_minus] ) * ( d_w_vec[ψ_minus] ) / ρ_w_vec[ψ_minus] 
    end    
    if ρ_w_vec[ψ_plus] > 0.0  
        D41_ψ_plus = ( GQ[ψ_plus] ) * ( gs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D42_ψ_plus = ( GQ[ψ_plus] ) * ( -bs[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D43_ψ_plus = ( GQ[ψ_plus] ) * ( GP[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        D44_ψ_plus = ( GQ[ψ_plus] ) * ( GQ[ψ_plus] ) / ρ_w_vec[ψ_plus]        
        d4_ψ_plus  = ( GQ[ψ_plus] ) * ( d_w_vec[ψ_plus] ) / ρ_w_vec[ψ_plus] 
    end      
      
    D41 = D41_ψ_minus + D41_ψ_plus;                                    
    D42 = D42_ψ_minus + D42_ψ_plus; 
    D43 = D43_ψ_minus + D43_ψ_plus; 
    D44 = D44_ψ_minus + D44_ψ_plus + ρ_q_vec[ψ_minus] + ρ_qg_vec[ψ_minus]; 
    d4  = d4_ψ_minus  + d4_ψ_plus  + d_q_vec[ψ_minus] - d_qg_vec[ψ_minus] + CQ[ψ_plus];
                                

    matrix_D = [
    D11 D12 D13 D14;
    D21 D22 D23 D24;
    D31 D32 D33 D34;
    D41 D42 D43 D44 ]
    d=[d1, d2, d3, d4]
 

    if d3 == 0.0 && d4 == 0.0 ## no line at one phase
        matrix_D = [
            D11 D12;
            D21 D22;
        ]
        d=[d1, d2]
        mu = matrix_D \ d
        μ_p = mu[1] 
        μ_q = mu[2]
        μ_p_psi_minus = 0.0
        μ_q_psi_minus = 0.0

    else
        mu = matrix_D \ d
        μ_p = mu[1] 
        μ_q = mu[2]
        μ_p_psi_minus = mu[3]
        μ_q_psi_minus = mu[4]

    end
 

    for (_,c) in enumerate(connections)    
        bus_idx = info.map_bus_idx[i,c]        
        temp = 0.0
        if c == ψ
            temp =  -(GP[c] + gs[c]) * μ_p - (GQ[c] - bs[c]) * μ_q
        end    
        if c == ψ_plus
            temp = - gs[c] * μ_p - GP[c] * μ_p_psi_minus + bs[c] * μ_q - GQ[c] * μ_q_psi_minus
        end    
        if c == ψ_minus
            temp = - gs[c] * μ_p_psi_minus - GP[c] * μ_p + bs[c] * μ_q_psi_minus - GQ[c] * μ_q
        end

        if ρ_w_vec[c] > 0.0              
            w[bus_idx] = ( d_w_vec[c] + temp ) / ρ_w_vec[c]                    
        else ## no arcs 
            if info.eta == Inf
                w[bus_idx] = temp
            else
                w[bus_idx] = param.w_prev[bus_idx] + info.eta*temp
            end
        end        
  
        for gen_idx in info.setG[i,c]
            idx = info.map_gen_idx[gen_idx,c]
            if c == ψ_minus 
                pg_hat[idx] = 1.0 / (param.ρ_pg[idx]+1.0/info.eta) * ( param.λ_pg[idx] + param.ρ_pg[idx]*param.pg[idx] + param.pg_hat_prev[idx]/info.eta + μ_p_psi_minus )
                qg_hat[idx] = 1.0 / (param.ρ_qg[idx]+1.0/info.eta) * ( param.λ_qg[idx] + param.ρ_qg[idx]*param.qg[idx] + param.qg_hat_prev[idx]/info.eta + μ_q_psi_minus )    
            else
                pg_hat[idx] = 1.0 / (param.ρ_pg[idx]+1.0/info.eta) * ( param.λ_pg[idx] + param.ρ_pg[idx]*param.pg[idx] + param.pg_hat_prev[idx]/info.eta + μ_p )
                qg_hat[idx] = 1.0 / (param.ρ_qg[idx]+1.0/info.eta) * ( param.λ_qg[idx] + param.ρ_qg[idx]*param.qg[idx] + param.qg_hat_prev[idx]/info.eta + μ_q )    
            end            
        end
        for line_idx in info.setA[i,c]
            idx = info.map_line_idx[line_idx,c]      
            if c == ψ_minus 
                p_hat[idx] = 1.0 / (param.ρ_p[idx]+1.0/info.eta) * ( param.λ_p[idx] + param.ρ_p[idx]*param.p[idx] + param.p_hat_prev[idx]/info.eta - μ_p_psi_minus )
                q_hat[idx] = 1.0 / (param.ρ_q[idx]+1.0/info.eta) * ( param.λ_q[idx] + param.ρ_q[idx]*param.q[idx] + param.q_hat_prev[idx]/info.eta - μ_q_psi_minus )
            else
                p_hat[idx] = 1.0 / (param.ρ_p[idx]+1.0/info.eta) * ( param.λ_p[idx] + param.ρ_p[idx]*param.p[idx] + param.p_hat_prev[idx]/info.eta - μ_p )
                q_hat[idx] = 1.0 / (param.ρ_q[idx]+1.0/info.eta) * ( param.λ_q[idx] + param.ρ_q[idx]*param.q[idx] + param.q_hat_prev[idx]/info.eta - μ_q )            
            end            
        end
    end
    return w, pg_hat, qg_hat, p_hat, q_hat 
end
 
function update_dual(pm, info, param, Line_info; nw=0)    
    param.λ_pg .+= param.ρ_pg .*(param.pg - param.pg_hat)    
    param.λ_qg .+= param.ρ_qg .*(param.qg - param.qg_hat)
    param.λ_p  .+= param.ρ_p .*(param.p - param.p_hat)
    param.λ_q  .+= param.ρ_q .*(param.q - param.q_hat)
 
    for line_idx = 1:info.NLines

        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]        
        
        for c in connections
            line_f_idx = info.map_line_idx[(l,i1,i2),c]
            line_t_idx = info.map_line_idx[(l,i2,i1),c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]

            param.λ_w[line_f_idx] += param.ρ_w[line_f_idx]*(param.w_hat[line_f_idx] - param.w[bus_f_idx])    
            param.λ_w[line_t_idx] += param.ρ_w[line_t_idx]*(param.w_hat[line_t_idx] - param.w[bus_t_idx])
        end
    end        
end
 
function residual_termination(pm, info, param, Line_info;nw=0)

    primal_res = sum( (param.pg - param.pg_hat).^2 ) + sum( (param.qg - param.qg_hat).^2 ) + sum( (param.p - param.p_hat).^2 )  + sum( (param.q - param.q_hat).^2 )
    dual_res = sum( (param.ρ_pg .*(param.pg_hat - param.pg_hat_prev)).^2 ) + sum( (param.ρ_qg .*(param.qg_hat - param.qg_hat_prev)).^2 ) + sum( (param.ρ_p .*(param.p_hat - param.p_hat_prev)).^2 ) + sum( (param.ρ_q .*(param.q_hat - param.q_hat_prev)).^2 )

    N_λ = 2*info.ngen + 6*(info.nbranch+info.ntrans)
    N_x = 2*info.ngen + 6*(info.nbranch+info.ntrans)

    ϵ_dual = sqrt(N_x)*info.ϵ_abs + info.ϵ_rel*sqrt(  sum( (param.λ_pg).^2 ) + sum( (param.λ_qg).^2 ) + sum( (param.λ_p).^2 )  + sum( (param.λ_q).^2 ) + sum( (param.λ_w).^2 )  )
    temp1 = sqrt( sum( (param.pg).^2 ) + sum( (param.qg).^2 ) + sum( (param.p).^2 )  + sum( (param.q).^2 ) + sum( (param.w_hat).^2 )  )
    temp2 = sum( (param.pg_hat).^2 ) + sum( (param.qg_hat).^2 ) + sum( (param.p_hat).^2 )  + sum( (param.q_hat).^2 )
    
    temp2_1 = 0
    for line_idx = 1:info.NLines
        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]        
        
        for c in connections
            line_f_idx = info.map_line_idx[(l,i1,i2),c]
            line_t_idx = info.map_line_idx[(l,i2,i1),c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]

            primal_res+=(param.w_hat[line_f_idx] - param.w[bus_f_idx])^2                             
            primal_res+=(param.w_hat[line_t_idx] - param.w[bus_t_idx])^2     
            
            dual_res+= param.ρ_w[line_f_idx]*(param.w[bus_f_idx] - param.w_prev[bus_f_idx])^2
            dual_res+= param.ρ_w[line_t_idx]*(param.w[bus_t_idx] - param.w_prev[bus_t_idx])^2

            temp2_1+=(param.w[bus_f_idx])^2 + (param.w[bus_t_idx])^2           
        end
    end
    temp3 = max( temp1, sqrt(temp2+temp2_1))    
    ϵ_prim = sqrt(N_λ)*info.ϵ_abs + info.ϵ_rel* temp3
 
    return sqrt(primal_res), sqrt(dual_res), ϵ_prim, ϵ_dual
end
 
function two_point_step_size(info, r_old, r_curr, r_next, ρ_curr, τ_avg)
    ϵ_rel = info.ϵ_rel
    ν_incr = info.ν_incr
    ν_decr = info.ν_decr
    σ = info.σ 
    ρ = ρ_curr
    
    if τ_avg >= ν_incr*ρ_curr && abs(r_next) > ϵ_rel &&  abs(r_curr) > ϵ_rel
        
        if abs(r_next) > σ*abs(r_old) || abs(r_curr) > σ*abs(r_old)
            ρ = ρ_curr*ν_incr
        end

    elseif τ_avg >= ρ_curr && abs(r_next) > ϵ_rel &&  abs(r_curr) > ϵ_rel 
        
        if abs(r_next) > σ*abs(r_old) || abs(r_curr) > σ*abs(r_old)
            ρ = τ_avg   
        end

    elseif τ_avg <= ρ_curr / ν_decr
        ρ = ρ_curr / ν_decr
    elseif τ_avg <= ρ_curr
        ρ = τ_avg   
    end
    ρ = clamp(ρ, info.ρ_min, info.ρ_max)    

    if ρ < info.ρ_min
        println("ρ=", ρ, " info.ρ_min=", info.ρ_min )
    end
    
    return ρ
end

function two_point_step_size(pm,info,param,Line_info, τ_pg_vector, τ_qg_vector, τ_p_vector, τ_q_vector, τ_w_vector; nw=0)
    
    τ_pg_avg = mean(τ_pg_vector, dims=1)[1]    
    τ_qg_avg = mean(τ_qg_vector, dims=1)[1]
    τ_p_avg  = mean(τ_p_vector, dims=1)[1]
    τ_q_avg  = mean(τ_q_vector, dims=1)[1]
    τ_w_avg  = mean(τ_w_vector, dims=1)[1]


    for (i,gen) in PMD.ref(pm, nw, :gen)        
        for c in gen["connections"]
            idx = info.map_gen_idx[i,c]
            param.ρ_pg[idx] = two_point_step_size(info, param.r_pg_old[idx], param.r_pg_curr[idx], param.r_pg_next[idx], param.ρ_pg[idx], τ_pg_avg[idx]) 
            param.ρ_qg[idx] = two_point_step_size(info, param.r_qg_old[idx], param.r_qg_curr[idx], param.r_qg_next[idx], param.ρ_qg[idx], τ_qg_avg[idx])             
        end
    end

    for line_idx = 1:info.NLines

        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]        
        
        for c in connections
            line_f_idx = info.map_line_idx[(l,i1,i2),c]
            line_t_idx = info.map_line_idx[(l,i2,i1),c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]

            param.ρ_p[line_f_idx] = two_point_step_size(info, param.r_p_old[line_f_idx], param.r_p_curr[line_f_idx], param.r_p_next[line_f_idx], param.ρ_p[line_f_idx], τ_p_avg[line_f_idx]) 
            param.ρ_q[line_f_idx] = two_point_step_size(info, param.r_q_old[line_f_idx], param.r_q_curr[line_f_idx], param.r_q_next[line_f_idx], param.ρ_q[line_f_idx], τ_q_avg[line_f_idx]) 
            param.ρ_w[line_f_idx] = two_point_step_size(info, param.r_w_old[line_f_idx], param.r_w_curr[line_f_idx], param.r_w_next[line_f_idx], param.ρ_w[line_f_idx], τ_w_avg[line_f_idx]) 
            
            param.ρ_p[line_t_idx] = two_point_step_size(info, param.r_p_old[line_t_idx], param.r_p_curr[line_t_idx], param.r_p_next[line_t_idx], param.ρ_p[line_t_idx], τ_p_avg[line_t_idx]) 
            param.ρ_q[line_t_idx] = two_point_step_size(info, param.r_q_old[line_t_idx], param.r_q_curr[line_t_idx], param.r_q_next[line_t_idx], param.ρ_q[line_t_idx], τ_q_avg[line_t_idx]) 
            param.ρ_w[line_t_idx] = two_point_step_size(info, param.r_w_old[line_t_idx], param.r_w_curr[line_t_idx], param.r_w_next[line_t_idx], param.ρ_w[line_t_idx], τ_w_avg[line_t_idx]) 
        end
    end 
end

function compute_tau(info, τ_prev, Δ_λ, Δ_y, Δ_z)
    ϵ_abs = info.ϵ_abs
    α = Δ_λ / Δ_y
    β = Δ_λ / Δ_z
    τ = 0.0    
    
    if Δ_y > ϵ_abs && Δ_z < ϵ_abs        
        τ = abs(α)
    elseif Δ_y < ϵ_abs && Δ_z > ϵ_abs        
        τ = abs(β)
    elseif Δ_y < ϵ_abs && Δ_z < ϵ_abs
        τ = τ_prev
    elseif Δ_λ < ϵ_abs
        τ = τ_prev
    else
        τ = sqrt( α*β )
    end
    
    return τ
end

function compute_tau(pm,info,param,Line_info;nw=0)      
   
    for (i,gen) in PMD.ref(pm, nw, :gen)        
        for c in gen["connections"]
            idx = info.map_gen_idx[i,c]

            Δ_λ = param.λ_pg[idx] - param.λ_pg_prev[idx]
            Δ_y = param.pg[idx] - param.pg_prev[idx]
            Δ_z = - param.pg_hat[idx] + param.pg_hat_prev[idx]            
            param.τ_pg[idx] = compute_tau(info, param.τ_pg[idx] , Δ_λ, Δ_y, Δ_z)

            Δ_λ = param.λ_qg[idx] - param.λ_qg_prev[idx]
            Δ_y = param.qg[idx] - param.qg_prev[idx]
            Δ_z = - param.qg_hat[idx] + param.qg_hat_prev[idx]            
            param.τ_qg[idx] = compute_tau(info, param.τ_qg[idx] , Δ_λ, Δ_y, Δ_z)             
        end
    end

    for line_idx = 1:info.NLines

        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]        
        
        for c in connections
            line_f_idx = info.map_line_idx[(l,i1,i2),c]
            line_t_idx = info.map_line_idx[(l,i2,i1),c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]

            # p
            Δ_λ = param.λ_p[line_f_idx] - param.λ_p_prev[line_f_idx]
            Δ_y = param.p[line_f_idx] - param.p_prev[line_f_idx]
            Δ_z = - param.p_hat[line_f_idx] + param.p_hat_prev[line_f_idx]            
            param.τ_p[line_f_idx] = compute_tau(info, param.τ_p[line_f_idx] , Δ_λ, Δ_y, Δ_z)

            Δ_λ = param.λ_p[line_t_idx] - param.λ_p_prev[line_t_idx]
            Δ_y = param.p[line_t_idx] - param.p_prev[line_t_idx]
            Δ_z = - param.p_hat[line_t_idx] + param.p_hat_prev[line_t_idx]            
            param.τ_p[line_t_idx] = compute_tau(info, param.τ_p[line_t_idx] , Δ_λ, Δ_y, Δ_z)

            # q
            Δ_λ = param.λ_q[line_f_idx] - param.λ_q_prev[line_f_idx]
            Δ_y = param.q[line_f_idx] - param.q_prev[line_f_idx]
            Δ_z = - param.q_hat[line_f_idx] + param.q_hat_prev[line_f_idx]            
            param.τ_q[line_f_idx] = compute_tau(info, param.τ_q[line_f_idx] , Δ_λ, Δ_y, Δ_z)

            Δ_λ = param.λ_q[line_t_idx] - param.λ_q_prev[line_t_idx]
            Δ_y = param.q[line_t_idx] - param.q_prev[line_t_idx]
            Δ_z = - param.q_hat[line_t_idx] + param.q_hat_prev[line_t_idx]            
            param.τ_q[line_t_idx] = compute_tau(info, param.τ_q[line_t_idx] , Δ_λ, Δ_y, Δ_z)

            # w
            Δ_λ = param.λ_w[line_f_idx] - param.λ_w_prev[line_f_idx]
            Δ_y = param.w_hat[line_f_idx] - param.w_hat_prev[line_f_idx]
            Δ_z = - param.w[bus_f_idx] + param.w_prev[bus_f_idx]            
            param.τ_w[line_f_idx] = compute_tau(info, param.τ_w[line_f_idx] , Δ_λ, Δ_y, Δ_z)

            Δ_λ = param.λ_w[line_t_idx] - param.λ_w_prev[line_t_idx]
            Δ_y = param.w_hat[line_t_idx] - param.w_hat_prev[line_t_idx]
            Δ_z = - param.w[bus_t_idx] + param.w_prev[bus_t_idx]            
            param.τ_w[line_t_idx] = compute_tau(info, param.τ_w[line_t_idx] , Δ_λ, Δ_y, Δ_z)
        end
    end 
end

function store_r_current_previous(pm,info,param, Line_info;nw=0)

    param.r_pg_curr .= param.pg_prev .- param.pg_hat_prev
    param.r_qg_curr .= param.qg_prev .- param.qg_hat_prev
    param.r_p_curr .= param.p_prev .- param.p_hat_prev
    param.r_q_curr .= param.q_prev .- param.q_hat_prev 

    param.r_pg_next .= param.pg  .- param.pg_hat 
    param.r_qg_next .= param.qg  .- param.qg_hat 
    param.r_p_next .= param.p  .- param.p_hat 
    param.r_q_next .= param.q  .- param.q_hat 
    for line_idx = 1:info.NLines

        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]        
        
        for c in connections
            line_f_idx = info.map_line_idx[(l,i1,i2),c]
            line_t_idx = info.map_line_idx[(l,i2,i1),c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]

            param.r_w_curr[line_f_idx] = param.w_hat_prev[line_f_idx] - param.w_prev[bus_f_idx]
            param.r_w_curr[line_t_idx] = param.w_hat_prev[line_t_idx] - param.w_prev[bus_t_idx]

            param.r_w_next[line_f_idx] = param.w_hat[line_f_idx] - param.w[bus_f_idx]
            param.r_w_next[line_t_idx] = param.w_hat[line_t_idx] - param.w[bus_t_idx] 
        end
    end     
end

function store_r_old(pm,info,param,Line_info;nw=0)
 
    param.r_pg_old .= param.pg  .- param.pg_hat 
    param.r_qg_old .= param.qg  .- param.qg_hat 
    param.r_p_old .= param.p  .- param.p_hat 
    param.r_q_old .= param.q  .- param.q_hat 

    for line_idx = 1:info.NLines

        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]        
        
        for c in connections
            line_f_idx = info.map_line_idx[(l,i1,i2),c]
            line_t_idx = info.map_line_idx[(l,i2,i1),c]
            bus_f_idx = info.map_bus_idx[i1,c]
            bus_t_idx = info.map_bus_idx[i2,c]

            param.r_w_old[line_f_idx] = param.w_hat[line_f_idx] - param.w[bus_f_idx]
            param.r_w_old[line_t_idx] = param.w_hat[line_t_idx] - param.w[bus_t_idx] 
        end
    end 
end
 
function solve_biased_bus(pm,info,param, biased_bus_model, var; nw=0)
    
    ## Biased_Bus_Model
    for (i,bus) in PMD.ref(pm, nw, :bus)                
        model = biased_bus_model[i]            
        org_obj_fn = JuMP.objective_function(model)   
   
        additional_term = 0
        for c in bus["terminals"]
            if info.PROX == "ON"
                idx = info.map_bus_idx[i,c]
                additional_term += (0.5/info.eta) * (var.biased_w[i,c] - param.w_prev[idx])^2                                       
            end    
            ## Generator
            if i in keys(info.map_bus_to_gen)
                gen_id = info.map_bus_to_gen[i]
                gen = PMD.ref(pm, nw, :gen, gen_id)                                                
                if c in gen["connections"]      
                    idx = info.map_gen_idx[gen_id,c]

                    term_1 = - param.λ_pg[idx]*var.biased_pg_hat[gen_id,c] - param.λ_qg[idx]*var.biased_qg_hat[gen_id,c]
                    term_2 = 0.5*param.ρ_pg[idx]*( param.pg[idx] - var.biased_pg_hat[gen_id,c] )^2 + 0.5*param.ρ_qg[idx]*( param.qg[idx] - var.biased_qg_hat[gen_id,c] )^2
                    
                    additional_term += term_1 + term_2

                    if info.PROX == "ON"
                        additional_term += (0.5/info.eta) * ( (var.biased_pg_hat[gen_id,c] - param.pg_hat_prev[idx])^2 
                                                            + (var.biased_qg_hat[gen_id,c] - param.qg_hat_prev[idx])^2 )
                    end         
                end                                                  
            end
            
            ## Transformer
            if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)                                                                 
                    if c in connections    
                        idx = info.map_line_idx[(l,i1,i2),c]

                        term_1 = - param.λ_p[idx]*var.biased_p_hat[(l,i1,i2),c] - param.λ_q[idx]*var.biased_q_hat[(l,i1,i2),c]  - param.λ_w[idx]*var.biased_w[i,c]
                        term_2 = 0.5*param.ρ_p[idx]*(param.p[idx] - var.biased_p_hat[(l,i1,i2),c])^2+ 0.5*param.ρ_q[idx]*(param.q[idx] - var.biased_q_hat[(l,i1,i2),c])^2 + 0.5*param.ρ_w[idx]*(param.w_hat[idx] - var.biased_w[i,c])^2
                        additional_term += term_1  + term_2

                        if info.PROX == "ON"
                            additional_term += (0.5/info.eta) * ( (var.biased_p_hat[(l,i1,i2),c] - param.p_hat_prev[idx])^2 
                                                                + (var.biased_q_hat[(l,i1,i2),c] - param.q_hat_prev[idx])^2 )
                        end                              
                    end 
                end             
            end
                        
            ## Branch
            if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)                                    
                    if c in connections                                                 
                        idx = info.map_line_idx[(l,i1,i2),c]

                        term_1 = - param.λ_p[idx]*var.biased_p_hat[(l,i1,i2),c] - param.λ_q[idx]*var.biased_q_hat[(l,i1,i2),c]  - param.λ_w[idx]*var.biased_w[i,c]                        
                        term_2 = 0.5*param.ρ_p[idx]*(param.p[idx] - var.biased_p_hat[(l,i1,i2),c])^2+ 0.5*param.ρ_q[idx]*(param.q[idx] - var.biased_q_hat[(l,i1,i2),c])^2 + 0.5*param.ρ_w[idx]*(param.w_hat[idx] - var.biased_w[i,c])^2
                        additional_term += term_1  + term_2

                        if info.PROX == "ON"
                            additional_term += (0.5/info.eta) * ( (var.biased_p_hat[(l,i1,i2),c] - param.p_hat_prev[idx])^2 
                                                                + (var.biased_q_hat[(l,i1,i2),c] - param.q_hat_prev[idx])^2 )
                        end  
                    end                                                                     
                end             
            end
        end 
         
        JuMP.set_objective_function(model, org_obj_fn + additional_term)                
        JuMP.optimize!(model)
                
        for c in bus["terminals"]
            bus_idx = info.map_bus_idx[i,c]
            param.biased_w[bus_idx] = JuMP.value( var.biased_w[i,c] )
            
            if i in keys(info.map_bus_to_gen)
                gen_id = info.map_bus_to_gen[i]
                gen = PMD.ref(pm, nw, :gen, gen_id)                            
                if c in gen["connections"]       
                    idx = info.map_gen_idx[gen_id,c]

                    param.biased_pg_hat[idx] = JuMP.value( var.biased_pg_hat[gen_id,c]  )
                    param.biased_qg_hat[idx] = JuMP.value( var.biased_qg_hat[gen_id,c]  )
                end
            end 
            if PMD.ref(pm, nw, :bus_arcs_conns_transformer, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)                                         
                    if c in connections    
                        idx = info.map_line_idx[(l,i1,i2),c]

                        param.biased_p_hat[idx] = JuMP.value( var.biased_p_hat[(l,i1,i2),c] )
                        param.biased_q_hat[idx] = JuMP.value( var.biased_q_hat[(l,i1,i2),c] )                        
                    end
                end
            end
            if PMD.ref(pm, nw, :bus_arcs_conns_branch, i) != []
                for ((l,i1,i2),connections) in PMD.ref(pm, nw, :bus_arcs_conns_branch, i)            
                    if c in connections                                                                         
                        idx = info.map_line_idx[(l,i1,i2),c]

                        param.biased_p_hat[idx] = JuMP.value( var.biased_p_hat[(l,i1,i2),c] )
                        param.biased_q_hat[idx] = JuMP.value( var.biased_q_hat[(l,i1,i2),c] )                        
                    end
                end
            end
        end            
        JuMP.set_objective_function(model, org_obj_fn)   
    end
 
    return param.biased_w, param.biased_pg_hat, param.biased_qg_hat, param.biased_p_hat, param.biased_q_hat
end

function ADMM_generate_laplace_noise(original, Δ, bar_ϵ)    
    for idx in keys(original)
        if Δ[idx] > 0.00001            
            rdnum = rand(Distributions.Laplace(0,Δ[idx]/bar_ϵ),1)[1]            
            original[idx] += rdnum            
        end        
    end    
    return original  
end
 
function ADMM_Serial(pm, info, param, Line_info, bus_model, biased_bus_model, var, res_io; nw=0)         
    Random.seed!(1)
    print_iteration_title(res_io)       
    a=1.0
    Kf=100
    Kt=10
    tmpcnt = 1
    τ_pg_vector = []
    τ_qg_vector = []
    τ_p_vector = []
    τ_q_vector = []
    τ_w_vector = []
    start_time = time()    
 
    for iteration = 1:info.TotalIteration  
        iter_start = time()
        
        ρ_mean = 0.2*( mean(param.ρ_pg) + mean(param.ρ_qg) + mean(param.ρ_p) + mean(param.ρ_q) + mean(param.ρ_w)  ) 
        
        if info.PROX == "P1"
            info.eta = a / sqrt(iteration)
        elseif info.PROX == "P2"
            info.eta = a / iteration
        elseif info.PROX == "P3"
            info.eta = a / (iteration)^2
        else
            info.eta = Inf
        end

        if info.TwoPSS == "ON" && (iteration % Kf) == 2   
            store_r_old(pm,info,param,Line_info)                       
        end  

        save_previous_solution_primal(param)  ## location important

        gen_start = time()
        solve_generator_closed_form(pm, info, param)        
        gen_elapsed = time()-gen_start

        line_start = time()
        solve_line(pm, info, param, Line_info, var)                 
        line_elapsed = time()-line_start
 
        bus_start = time()
        if info.ClosedForm == "ON"
            param.w, param.pg_hat, param.qg_hat, param.p_hat, param.q_hat = solve_bus_closed_form(pm, info, param, var, 1.0)        
        else
            solve_bus(pm, info, param, bus_model, var; nw=0)        
        end        
        bus_elapsed = time()-bus_start
        
        if info.bar_ϵ != Inf            
            
            if info.ClosedForm == "ON"                                       
                param.biased_w, param.biased_pg_hat, param.biased_qg_hat, param.biased_p_hat, param.biased_q_hat = solve_bus_closed_form(pm, info, param, var, info.β)
            else
                param.biased_w, param.biased_pg_hat, param.biased_qg_hat, param.biased_p_hat, param.biased_q_hat = solve_biased_bus(pm,info,param,biased_bus_model, var)      
            end

            ## Δ Calculation
            Δ_pg_hat    = abs.(param.biased_pg_hat - param.pg_hat ) 
            Δ_qg_hat    = abs.(param.biased_qg_hat - param.qg_hat ) 
            Δ_p_hat     = abs.(param.biased_p_hat - param.p_hat ) 
            Δ_q_hat     = abs.(param.biased_q_hat - param.q_hat ) 
            Δ_w         = abs.(param.biased_w - param.w ) 

            info.Δ_avg = 0.2*( mean(Δ_pg_hat)+mean(Δ_qg_hat)+mean(Δ_p_hat)+mean(Δ_q_hat)+mean(Δ_w)  )
            
            param.pg_hat = ADMM_generate_laplace_noise(param.pg_hat, Δ_pg_hat, info.bar_ϵ)
            param.qg_hat = ADMM_generate_laplace_noise(param.qg_hat, Δ_qg_hat, info.bar_ϵ)
            param.p_hat  = ADMM_generate_laplace_noise(param.p_hat, Δ_p_hat, info.bar_ϵ)
            param.q_hat  = ADMM_generate_laplace_noise(param.q_hat, Δ_q_hat, info.bar_ϵ)
            param.w      = ADMM_generate_laplace_noise(param.w, Δ_w, info.bar_ϵ)                
        end
 
        if info.TwoPSS == "ON"
            ## compute τ (location important)
            compute_tau(pm,info,param,Line_info)                    
            if iteration >= tmpcnt*Kf - Kt + 1 && iteration <= tmpcnt*Kf
                push!(τ_pg_vector, param.τ_pg)
                push!(τ_qg_vector, param.τ_qg)
                push!(τ_p_vector, param.τ_p)
                push!(τ_q_vector, param.τ_q)
                push!(τ_w_vector, param.τ_w)
            end
        end

        save_previous_solution_dual(param) ## location important
        dual_start = time()
        update_dual(pm, info, param, Line_info)
        dual_elapsed = time()-dual_start        
        primal_res, dual_res, ϵ_prim, ϵ_dual = residual_termination(pm, info, param, Line_info)

        if info.TwoPSS == "ON" && (iteration % Kf) == 0            
            store_r_current_previous(pm,info,param, Line_info)
            two_point_step_size(pm,info,param,Line_info, τ_pg_vector, τ_qg_vector, τ_p_vector, τ_q_vector, τ_w_vector)                        
            ## Initialize
            param.τ_pg = zeros(info.ngen)
            param.τ_qg = zeros(info.ngen)
            param.τ_p = zeros(2*(info.nbranch+info.ntrans))
            param.τ_q = zeros(2*(info.nbranch+info.ntrans))
            param.τ_w = zeros(2*(info.nbranch+info.ntrans))
            τ_pg_vector = []
            τ_qg_vector = []
            τ_p_vector = []
            τ_q_vector = []
            τ_w_vector = []
            tmpcnt += 1            
        end
        
        iter_elapsed = time() - iter_start
        elapsed_time = time() - start_time

        if (iteration % info.display_step) == 0
            print_iteration(res_io, param.pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, info.Δ_avg, elapsed_time, gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)     
        end
  
        if iteration == info.TotalIteration  
            print_iteration(res_io, param.pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, info.Δ_avg, elapsed_time, gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)     
            @printf(res_io, "\n")    
            @printf(res_io, "Iteration Limit")    
            print_summary(param.pg_hat, info.bar_ϵ, res_io)     
            break;
        end

        if primal_res <= ϵ_prim && dual_res <= ϵ_dual
            print_iteration(res_io, param.pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, info.Δ_avg, elapsed_time, gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)     
            @printf(res_io, "\n")    
            @printf(res_io, "Termination")    
            print_summary(param.pg_hat, info.bar_ϵ, res_io)     
            break;
        end            
    end
end 

########################################################
#### Parallel
########################################################
function solve_line_parallel(var, Line_info, line_in_vec, line_out_vec, Line_in_map_idx, Line_out_map_idx, NLines_start, NLines_last)
    ## Solve line
    for line_idx = NLines_start:NLines_last

        model       = Line_info[line_idx][3]
        l           = Line_info[line_idx][2]
        i1          = Line_info[line_idx][4]
        i2          = Line_info[line_idx][5]
        connections = Line_info[line_idx][6]   
        
        f_idx = (l,i1,i2)
        t_idx = (l,i2,i1)

        org_obj_fn = JuMP.objective_function(model)            
        
        additional_term = 0
        for c in connections 

            ρ_p_f = line_in_vec[ Line_in_map_idx[line_idx, c, 1] ] 
            ρ_p_t = line_in_vec[ Line_in_map_idx[line_idx, c, 2] ] 
            ρ_q_f = line_in_vec[ Line_in_map_idx[line_idx, c, 3] ] 
            ρ_q_t = line_in_vec[ Line_in_map_idx[line_idx, c, 4] ] 
            ρ_w_f = line_in_vec[ Line_in_map_idx[line_idx, c, 5] ] 
            ρ_w_t = line_in_vec[ Line_in_map_idx[line_idx, c, 6] ] 
            λ_p_f = line_in_vec[ Line_in_map_idx[line_idx, c, 7] ] 
            λ_p_t = line_in_vec[ Line_in_map_idx[line_idx, c, 8] ] 
            λ_q_f = line_in_vec[ Line_in_map_idx[line_idx, c, 9] ] 
            λ_q_t = line_in_vec[ Line_in_map_idx[line_idx, c, 10] ] 
            λ_w_f = line_in_vec[ Line_in_map_idx[line_idx, c, 11] ] 
            λ_w_t = line_in_vec[ Line_in_map_idx[line_idx, c, 12] ] 
            p_hat_f = line_in_vec[ Line_in_map_idx[line_idx, c, 13] ] 
            p_hat_t = line_in_vec[ Line_in_map_idx[line_idx, c, 14] ] 
            q_hat_f = line_in_vec[ Line_in_map_idx[line_idx, c, 15] ] 
            q_hat_t = line_in_vec[ Line_in_map_idx[line_idx, c, 16] ] 
            w_f     = line_in_vec[ Line_in_map_idx[line_idx, c, 17] ] 
            w_t     = line_in_vec[ Line_in_map_idx[line_idx, c, 18] ] 
 
            term_1 = λ_p_f * var.p[f_idx,c] + λ_p_t * var.p[t_idx,c] + λ_q_f * var.q[f_idx,c] + λ_q_t * var.q[t_idx,c] + λ_w_f * var.w_hat[f_idx,c] + λ_w_t * var.w_hat[t_idx,c]         
            term_1_rho = 0.5*ρ_p_f*( var.p[f_idx,c] - p_hat_f )^2 + 0.5*ρ_p_t*( var.p[t_idx,c] - p_hat_t )^2 + 0.5*ρ_q_f*( var.q[f_idx,c] - q_hat_f )^2 + 0.5*ρ_q_t*( var.q[t_idx,c] - q_hat_t )^2 + 0.5*ρ_w_f*( var.w_hat[f_idx,c] - w_f )^2  + 0.5*ρ_w_t*( var.w_hat[t_idx,c] - w_t )^2
            additional_term += term_1 + term_1_rho 
              
        end            
                
        JuMP.set_objective_function(model, org_obj_fn + additional_term)                        
        JuMP.optimize!(model)        
        if JuMP.termination_status(model) != OPTIMAL
            println("Status (trans)=", JuMP.termination_status(model))
            println(model)
        end 
        
        for c in connections     
            line_out_vec[ Line_out_map_idx[line_idx, c, 1] ]  = JuMP.value( var.p[f_idx,c]  )
            line_out_vec[ Line_out_map_idx[line_idx, c, 2] ]  = JuMP.value( var.p[t_idx,c]  )
            line_out_vec[ Line_out_map_idx[line_idx, c, 3] ]  = JuMP.value( var.q[f_idx,c]  )
            line_out_vec[ Line_out_map_idx[line_idx, c, 4] ]  = JuMP.value( var.q[t_idx,c]  )
            line_out_vec[ Line_out_map_idx[line_idx, c, 5] ]  = JuMP.value( var.w_hat[f_idx,c] )
            line_out_vec[ Line_out_map_idx[line_idx, c, 6] ]  = JuMP.value( var.w_hat[t_idx,c] )
        end

        JuMP.set_objective_function(model, org_obj_fn)       
    end 

    return line_out_vec
end 

function ADMM_Master(pm, info, Line_info, var, param, NLines_counts, res_io, 
    NLines_start, NLines_last, line_in_size, line_out_size, line_in_map_idx, line_out_map_idx)
     
    Lines_in_counts, Lines_out_counts = compute_Lines_counts(NLines_counts, line_in_size, line_out_size)
        
    print_iteration_title(res_io)       
          
    a=1.0
    Kf=100
    Kt=10
    tmpcnt = 1
    τ_pg_vector = []
    τ_qg_vector = []
    τ_p_vector = []
    τ_q_vector = []
    τ_w_vector = []
    start_time = time()     
    
    for iteration = 1:info.TotalIteration  

        iter_start = time()
        ## Send termination status to worker             
        MPI.Bcast!(status, 0, comm)
        
        if (status[1] == 1)
            break;
        end

        ρ_mean = 0.2*( mean(param.ρ_pg) + mean(param.ρ_qg) + mean(param.ρ_p) + mean(param.ρ_q) + mean(param.ρ_w)  ) 

        if info.PROX == "P1"
            info.eta = a / sqrt(iteration)
        elseif info.PROX == "P2"
            info.eta = a / iteration
        elseif info.PROX == "P3"
            info.eta = a / (iteration)^2
        else
            info.eta = Inf
        end
          
        if info.TwoPSS == "ON" && (iteration % Kf) == 2                 
            store_r_old(pm,info,param,Line_info)  
        end 
 
        save_previous_solution_primal(param)  ## location important
        gen_start = time()
        solve_generator_closed_form(pm, info, param)        
        gen_elapsed = time()-gen_start

        line_start = time()
        
        ## Construct Lines vectors
        Lines_in_vector, Lines_out_vector = construct_Lines_vector(info, param, Line_info)

        ## SEND
        MPI.Scatterv!(VBuffer(Lines_in_vector, Lines_in_counts), MPI.IN_PLACE, root, comm)
        
        ## RECEIVE
        MPI.Gatherv!(MPI.IN_PLACE, VBuffer(Lines_out_vector, Lines_out_counts), root, comm)
 
        ## UPDATE      
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

                param.p[line_f_idx] = Lines_out_vector[ line_out_map_idx[line_idx, c, 1] ]
                param.p[line_t_idx] = Lines_out_vector[ line_out_map_idx[line_idx, c, 2] ]
                param.q[line_f_idx] = Lines_out_vector[ line_out_map_idx[line_idx, c, 3] ]
                param.q[line_t_idx] = Lines_out_vector[ line_out_map_idx[line_idx, c, 4] ]
                param.w_hat[line_f_idx] = Lines_out_vector[ line_out_map_idx[line_idx, c, 5] ]
                param.w_hat[line_t_idx] = Lines_out_vector[ line_out_map_idx[line_idx, c, 6] ]
            end
        end
         
        line_elapsed = time()-line_start

        bus_start = time()
        param.w, param.pg_hat, param.qg_hat, param.p_hat, param.q_hat = solve_bus_closed_form(pm, info, param, var, 1.0)  
        bus_elapsed = time()-bus_start 
         
        if info.TwoPSS == "ON"
            ## compute τ (location important)
            compute_tau(pm,info,param,Line_info)                    
            if iteration >= tmpcnt*Kf - Kt + 1 && iteration <= tmpcnt*Kf
                push!(τ_pg_vector, param.τ_pg)
                push!(τ_qg_vector, param.τ_qg)
                push!(τ_p_vector, param.τ_p)
                push!(τ_q_vector, param.τ_q)
                push!(τ_w_vector, param.τ_w)
            end
        end
          
        save_previous_solution_dual(param) ## location important
        dual_start = time()
        update_dual(pm, info, param, Line_info)
        dual_elapsed = time()-dual_start
           
        primal_res, dual_res, ϵ_prim, ϵ_dual = residual_termination(pm, info, param, Line_info)
        
         
        if info.TwoPSS == "ON"
            if (iteration % Kf) == 0                
                store_r_current_previous(pm,info,param, Line_info)    
                two_point_step_size(pm,info,param, Line_info, τ_pg_vector, τ_qg_vector, τ_p_vector, τ_q_vector, τ_w_vector)            
                
                ## Initialize
                param.τ_pg = zeros(info.ngen)
                param.τ_qg = zeros(info.ngen)
                param.τ_p = zeros(2*(info.nbranch+info.ntrans))
                param.τ_q = zeros(2*(info.nbranch+info.ntrans))
                param.τ_w = zeros(2*(info.nbranch+info.ntrans))
                τ_pg_vector = []
                τ_qg_vector = []
                τ_p_vector = []
                τ_q_vector = []
                τ_w_vector = []
                tmpcnt += 1
            end
        end
        iter_elapsed = time() - iter_start
        elapsed_time = time() - start_time
        if (iteration % info.display_step) == 0            
            print_iteration(res_io, param.pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, info.Δ_avg, elapsed_time, 
            gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)     
        end
  
        if iteration == info.TotalIteration  
            print_iteration(res_io, param.pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, info.Δ_avg, elapsed_time, 
            gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)     
            @printf(res_io, "\n")    
            @printf(res_io, "Iteration Limit")    
            print_summary(param.pg_hat, info.bar_ϵ, res_io)     
            status[1] = 1
        end

        if primal_res <= ϵ_prim && dual_res <= ϵ_dual
            print_iteration(res_io, param.pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, info.Δ_avg, elapsed_time, 
            gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)     
            @printf(res_io, "\n")    
            @printf(res_io, "Termination")    
            print_summary(param.pg_hat, info.bar_ϵ, res_io)                 
            status[1] = 1
        end            
    end
    println("END: master $(rank)")
 
end

function ADMM_Sub(pm, info, Line_info, var, NLines_start, NLines_last, Line_in_size, Line_out_size, Line_in_map_idx, Line_out_map_idx)
    while true
        MPI.Bcast!(status, 0, comm)        
        if (status[1] == 1)
            break;
        end
 
        line_in_vec = zeros(Line_in_size)
        line_out_vec = zeros(Line_out_size)
        ## RECEIVE
        MPI.Scatterv!(nothing, line_in_vec, root, comm)   
        
        ## SOLVE
        line_out_vec = solve_line_parallel(var, Line_info, line_in_vec, line_out_vec, Line_in_map_idx, Line_out_map_idx, NLines_start, NLines_last)

        ## SEND
        MPI.Gatherv!(line_out_vec, nothing, root, comm)

    end
    println("END: sub $(rank)")
end





function ADMM_Serial_TEST(pm, info, param, Line_info, bus_model, biased_bus_model, var, res_io; nw=0)         
    Random.seed!(1)
    
      
    a=1.0
    Kf=100
    Kt=10
    tmpcnt = 1
    τ_pg_vector = []
    τ_qg_vector = []
    τ_p_vector = []
    τ_q_vector = []
    τ_w_vector = []
    start_time = time()    
 
    for iteration = 1:info.TotalIteration  
        iter_start = time()
        
        ρ_mean = 0.2*( mean(param.ρ_pg) + mean(param.ρ_qg) + mean(param.ρ_p) + mean(param.ρ_q) + mean(param.ρ_w)  ) 
        
        if info.PROX == "P1"
            info.eta = a / sqrt(iteration)
        elseif info.PROX == "P2"
            info.eta = a / iteration
        elseif info.PROX == "P3"
            info.eta = a / (iteration)^2
        else
            info.eta = Inf
        end 

        save_previous_solution_primal(param)  ## location important

        gen_start = time()
        solve_generator_closed_form(pm, info, param)        
        gen_elapsed = time()-gen_start

        line_start = time()
        solve_line(pm, info, param, Line_info, var)                 
        line_elapsed = time()-line_start
 
        bus_start = time()
        if info.ClosedForm == "ON"
            param.w, param.pg_hat, param.qg_hat, param.p_hat, param.q_hat = solve_bus_closed_form(pm, info, param, var, 1.0)        
        else
            solve_bus(pm, info, param, bus_model, var; nw=0)        
        end        
        bus_elapsed = time()-bus_start
         

        save_previous_solution_dual(param) ## location important
        dual_start = time()
        update_dual(pm, info, param, Line_info)
        dual_elapsed = time()-dual_start        
        primal_res, dual_res, ϵ_prim, ϵ_dual = residual_termination(pm, info, param, Line_info)
 
        iter_elapsed = time() - iter_start
        elapsed_time = time() - start_time

        if (iteration % info.display_step) == 0
            print_iteration(res_io, param.pg_hat, iteration, ρ_mean, primal_res, dual_res, ϵ_prim, ϵ_dual, info.Δ_avg, elapsed_time, gen_elapsed,line_elapsed, bus_elapsed, dual_elapsed, iter_elapsed)     
        end
     
    end
end