mutable struct Variables
    pd::Dict{}
    qd::Dict{}

    pg::Dict{}
    qg::Dict{}
    pg_hat::Dict{}
    qg_hat::Dict{}
    p::Dict{}
    q::Dict{}
    p_hat::Dict{}
    q_hat::Dict{}
    w::Dict{}
    w_hat::Dict{}

    biased_pd::Dict{}
    biased_qd::Dict{}
    biased_pg_hat::Dict{}
    biased_qg_hat::Dict{}
    biased_p_hat::Dict{}
    biased_q_hat::Dict{}
    biased_w::Dict{}
 
    function Variables()
        var = new()
        var.pd=Dict{}()
        var.qd=Dict{}()
        var.pg=Dict{}()
        var.qg=Dict{}()
        var.pg_hat=Dict{}()
        var.qg_hat=Dict{}()
        var.p=Dict{}()
        var.q=Dict{}()
        var.p_hat=Dict{}()
        var.q_hat=Dict{}()
        var.w=Dict{}()
        var.w_hat=Dict{}()    

        var.biased_pd=Dict{}()    
        var.biased_qd=Dict{}()    
        var.biased_pg_hat=Dict{}()    
        var.biased_qg_hat=Dict{}()    
        var.biased_p_hat=Dict{}()    
        var.biased_q_hat=Dict{}()    
        var.biased_w=Dict{}()    
 
        return var
    end
end

mutable struct Info
    Instance::String
    ALGO::String
    PROX::String
    TwoPSS::String
    ClosedForm::String
    
    Delta_bus::Vector{}
    Special_Delta_bus::Vector{}
    Wye_bus::Vector{}

    map_bus_to_gen::Dict{}
    map_bus_to_load::Dict{}    
    map_bus_idx::Dict{}
    map_gen_idx::Dict{}
    map_line_idx::Dict{}
    line_info::Dict{}
    setA::Dict{}
    setG::Dict{}
    ngen::Int64
    ntrans::Int64
    nbranch::Int64
    nbus::Int64
    NLines::Int64
    NBuses::Int64    
    
    TotalIteration::Int64
    display_step::Int64

    Initial_ρ::Float64
    mpq::Float64    
    mw::Float64

    ϵ_abs::Float64
    ϵ_rel::Float64

    bar_ϵ::Float64
    β::Float64
    δ::Float64
    Δ_avg::Float64
    solve_biased::Int64
    
    π_init::Float64
    ν_incr::Float64
    ν_decr::Float64
    σ::Float64
    ρ_max::Float64
    ρ_min::Float64

    eta::Float64
    
    tempbusid::Int64
    
    function Info()
        info = new()
        info.map_bus_to_gen=Dict{}()         
        info.map_bus_to_load=Dict{}()                 
        info.map_bus_idx=Dict{}()         
        info.map_gen_idx=Dict{}()         
        info.map_line_idx=Dict{}()   
        info.line_info=Dict{}()
        info.setA=Dict()
        info.setG=Dict()
        info.ngen=0
        info.ntrans=0
        info.nbranch=0
        info.nbus=0
        info.NLines=0
        info.NBuses=0  

        info.TotalIteration = 10000
        info.display_step   = 1  
        
        info.Initial_ρ      = 10.0
        info.mpq            = 0.0        
        info.mw             = 0.0

        info.ϵ_abs          = 1e-6   ## ϵ_abs  1e-6, 1e-4
        info.ϵ_rel          = 1e-5   ## ϵ_rel  5e-5, 1e-2, 1e-3 or 1e-4

        info.bar_ϵ          = Inf
        info.β              = 1.1
        info.δ              = 0.0
        info.Δ_avg          = 0.0
        info.solve_biased   = 1
        
        info.π_init         = 0.0
        info.ν_incr         = 2.0
        info.ν_decr         = 2.0
        info.σ              = 0.01
        info.ρ_max          = 1e6
        info.ρ_min          = 1e-3
        
        info.eta            = Inf
        
        info.tempbusid = 100000    # IEEE13: 1, 14, 16  IEEE34: 25, 40, 44, 49
        
        return info
    end
end

mutable struct OptimalPoints
    
    pg::Vector{}
    qg::Vector{}
    pg_hat::Vector{}
    qg_hat::Vector{}
    
    p::Vector{}
    q::Vector{}    
    p_hat::Vector{}
    q_hat::Vector{}         
    
    w::Vector{}
    w_hat::Vector{}

    λ_pg::Vector{}
    λ_qg::Vector{}
    λ_p::Vector{}
    λ_q::Vector{}
    λ_w::Vector{}

    function OptimalPoints(info)
        optimal = new()

        optimal.pg = zeros(info.ngen)
        optimal.qg = zeros(info.ngen)        
        optimal.p = zeros(2*(info.nbranch+info.ntrans))
        optimal.q = zeros(2*(info.nbranch+info.ntrans))
        optimal.w = ones(info.nbus)
         
        optimal.pg_hat = zeros(info.ngen)
        optimal.qg_hat = zeros(info.ngen)                
        optimal.p_hat = zeros(2*(info.nbranch+info.ntrans))
        optimal.q_hat = zeros(2*(info.nbranch+info.ntrans))
        optimal.w_hat = ones(2*(info.nbranch+info.ntrans))

        optimal.λ_pg = zeros(info.ngen)
        optimal.λ_qg = zeros(info.ngen)
        optimal.λ_p = zeros(2*(info.nbranch+info.ntrans))
        optimal.λ_q = zeros(2*(info.nbranch+info.ntrans))
        optimal.λ_w = zeros(2*(info.nbranch+info.ntrans))
        return optimal
    end
end
 
mutable struct ADMMParameter
    
    # generator (ngen)    
    pg::Vector{}
    qg::Vector{}
    pg_hat::Vector{}
    qg_hat::Vector{}
    λ_pg::Vector{}
    λ_qg::Vector{}

    #line ( 2*(nbranch+ntrans) )
    p::Vector{}
    q::Vector{}    
    p_hat::Vector{}
    q_hat::Vector{}
    λ_p::Vector{}
    λ_q::Vector{}
     
    #bus (nbus)
    w::Vector{}

    #bus at line (  2*(nbranch+ntrans)  )
    w_hat::Vector{}
    λ_w::Vector{}

    # penalty parameter
    ρ_pg::Vector{}
    ρ_qg::Vector{}
    ρ_p::Vector{}
    ρ_q::Vector{}
    ρ_w::Vector{}

    # previous
    pg_hat_prev::Vector{}
    qg_hat_prev::Vector{}
    p_hat_prev::Vector{}
    q_hat_prev::Vector{}
    w_prev::Vector{}

    λ_pg_prev::Vector{}
    λ_qg_prev::Vector{}
    λ_p_prev::Vector{}
    λ_q_prev::Vector{}
    λ_w_prev::Vector{}

    pg_prev::Vector{}
    qg_prev::Vector{}
    p_prev::Vector{}
    q_prev::Vector{}
    w_hat_prev::Vector{}

    r_pg_old::Vector{}
    r_qg_old::Vector{}
    r_p_old::Vector{}
    r_q_old::Vector{}
    r_w_old::Vector{}
    
    r_pg_curr::Vector{}
    r_qg_curr::Vector{}
    r_p_curr::Vector{}
    r_q_curr::Vector{}
    r_w_curr::Vector{}

    r_pg_next::Vector{}
    r_qg_next::Vector{}
    r_p_next::Vector{}
    r_q_next::Vector{}
    r_w_next::Vector{}

    τ_pg::Vector{}
    τ_qg::Vector{}
    τ_p::Vector{}
    τ_q::Vector{}
    τ_w::Vector{}

    biased_pg_hat::Vector{}
    biased_qg_hat::Vector{}
    biased_p_hat::Vector{}
    biased_q_hat::Vector{}
    biased_w::Vector{}

    function ADMMParameter(info)
        param = new()

        param.τ_pg = zeros(info.ngen)
        param.τ_qg = zeros(info.ngen)
        param.τ_p = zeros(2*(info.nbranch+info.ntrans))
        param.τ_q = zeros(2*(info.nbranch+info.ntrans))
        param.τ_w = zeros(2*(info.nbranch+info.ntrans))
         
        param.r_pg_old = zeros(info.ngen)
        param.r_qg_old = zeros(info.ngen)
        param.r_p_old = zeros(2*(info.nbranch+info.ntrans))
        param.r_q_old = zeros(2*(info.nbranch+info.ntrans))
        param.r_w_old = zeros(2*(info.nbranch+info.ntrans))

        param.r_pg_curr = zeros(info.ngen)
        param.r_qg_curr = zeros(info.ngen)
        param.r_p_curr = zeros(2*(info.nbranch+info.ntrans))
        param.r_q_curr = zeros(2*(info.nbranch+info.ntrans))
        param.r_w_curr = zeros(2*(info.nbranch+info.ntrans))
    
        param.r_pg_next = zeros(info.ngen)
        param.r_qg_next = zeros(info.ngen)
        param.r_p_next = zeros(2*(info.nbranch+info.ntrans))
        param.r_q_next = zeros(2*(info.nbranch+info.ntrans))
        param.r_w_next = zeros(2*(info.nbranch+info.ntrans))

        param.λ_pg_prev = zeros(info.ngen)
        param.λ_qg_prev = zeros(info.ngen)
        param.λ_p_prev = zeros(2*(info.nbranch+info.ntrans))
        param.λ_q_prev = zeros(2*(info.nbranch+info.ntrans))        
        param.λ_w_prev = zeros(2*(info.nbranch+info.ntrans))

        param.pg_hat_prev = zeros(info.ngen)
        param.qg_hat_prev = zeros(info.ngen)
        param.p_hat_prev = zeros(2*(info.nbranch+info.ntrans))
        param.q_hat_prev = zeros(2*(info.nbranch+info.ntrans))
        param.w_prev = zeros(info.nbus)

        param.pg_prev = zeros(info.ngen)
        param.qg_prev = zeros(info.ngen)
        param.p_prev = zeros(2*(info.nbranch+info.ntrans))
        param.q_prev = zeros(2*(info.nbranch+info.ntrans))
        param.w_hat_prev =zeros(2*(info.nbranch+info.ntrans))
        
 
        param.ρ_pg = zeros(info.ngen)
        param.ρ_qg = zeros(info.ngen)
        param.ρ_p = zeros(2*(info.nbranch+info.ntrans))
        param.ρ_q = zeros(2*(info.nbranch+info.ntrans))
        param.ρ_w = zeros(2*(info.nbranch+info.ntrans))


        param.pg = zeros(info.ngen)
        param.qg = zeros(info.ngen)
        param.pg_hat = zeros(info.ngen)
        param.qg_hat = zeros(info.ngen)
        param.λ_pg = zeros(info.ngen)
        param.λ_qg = zeros(info.ngen)

        param.p = zeros(2*(info.nbranch+info.ntrans))
        param.q = zeros(2*(info.nbranch+info.ntrans))
        param.p_hat = zeros(2*(info.nbranch+info.ntrans))
        param.q_hat = zeros(2*(info.nbranch+info.ntrans))
        param.λ_p = zeros(2*(info.nbranch+info.ntrans))
        param.λ_q = zeros(2*(info.nbranch+info.ntrans))
        param.w = ones(info.nbus)
        param.w_hat =zeros(2*(info.nbranch+info.ntrans))
        param.λ_w = zeros(2*(info.nbranch+info.ntrans))


        param.biased_pg_hat = zeros(info.ngen)
        param.biased_qg_hat = zeros(info.ngen)        
        param.biased_p_hat = zeros(2*(info.nbranch+info.ntrans))
        param.biased_q_hat = zeros(2*(info.nbranch+info.ntrans))
        param.biased_w = zeros(info.nbus)

        return param
    end
end