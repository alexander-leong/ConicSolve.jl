# see CVXOPT Vandenberghe 2010 for Cone QP definition

include("./arrayutils.jl")
include("./debug.jl")
include("./kktsolver.jl")
include("./cones/cone.jl")

# using BenchmarkTools
using LinearAlgebra
using Logging
using PrettyTables

@enum TerminationStatus begin
    ALMOST_OPTIMAL
    DUAL_INFEASIBLE
    INFEASIBLE
    ITERATION_LIMIT
    NORM_LIMIT
    NUMERICAL_ERROR
    OBJECTIVE_LIMIT
    OPTIMIZE_NOT_CALLED
    OTHER_ERROR
    OTHER_LIMIT
    OPTIMAL
    SLOW_PROGRESS
    TIME_LIMIT
end

@enum ConeTypes begin
    NONNEGATIVE_ORTHANT
    POSITIVE_SEMIDEFINITE_CONE
    SECOND_ORDER_CONE
end

mutable struct ConeQP
    A::Matrix{Float64}
    G::Matrix{Float64}
    KKT_A::Matrix{Float64}
    KKT_b::Vector{Float64}
    KKT_x::Vector{Float64}
    P::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    h::Vector{Float64}
    s::Vector{Float64}
    z::Vector{Float64}
    α::Float64
    
    cones::Vector{Cone}
    cones_inds::Vector{Int}
    cones_p::Vector{Int}

    inds_c::UnitRange{Int64}
    inds_b::UnitRange{Int64}
    inds_h::UnitRange{Int64}

    function ConeQP(A::Matrix{Float64},
                    G::Matrix{Float64},
                    P::Matrix{Float64},
                    b::Vector{Float64},
                    c::Vector{Float64},
                    h::Vector{Float64},
                    cones::Vector{Cone})
        cone_qp = new()
        cone_qp.A = A
        cone_qp.G = G
        cone_qp.P = P
        cone_qp.b = b
        cone_qp.c = c
        cone_qp.h = h
        cone_qp.cones = cones
        
        cone_qp.inds_c = 1:size(P)[1]
        cone_qp.inds_b = cone_qp.inds_c[end]+1:cone_qp.inds_c[end]+size(A)[1]
        cone_qp.inds_h = cone_qp.inds_b[end]+1:cone_qp.inds_b[end]+size(G)[1]
        return cone_qp
    end
end

mutable struct Solver
    current_iteration::Int32
    device # TODO: what to do?
    limit_obj::Union{Float32, Nothing}
    limit_soln::Union{Float32, Nothing}
    log_level
    max_iterations::Union{Int8, Nothing}
    num_threads::Union{Int8, Nothing}
    obj_dual_value::Float32
    obj_primal_value::Float32
    program::ConeQP
    solve_time
    status_dual
    status_primal
    status_termination::Union{TerminationStatus, Nothing}
    time_limit_sec::Int32
    tol_gap_abs::Union{Float32, Nothing}
    tol_gap_rel::Union{Float32, Nothing}
    tol_optimality::Union{Float32, Nothing}

    function Solver(program::ConeQP)
        solver = new()
        solver.current_iteration = 1
        solver.program = program
        return solver
    end
end

function getVariablePrimal(solver::Solver)
    return solver.program.KKT_x
end

function print_graphic()
    fp = open("graphic.txt", "r")
    words = readlines(fp, keep=true)
    for word in words
        print(word)
    end
end

function optimize!(solver::Solver)
    try
        print_graphic()
        @info "Optimize called"
        primal_obj = initialize!(solver)
        solver.obj_primal_value = primal_obj
        check_preconditions(solver)
        optimize_main!(solver)
        log_msg = ("Solver finished" * "\n" *
            "Exit status: " * string(solver.status_termination) * "\n" *
            "Primal objective value: " * string(solver.obj_primal_value) * "\n" *
            "Number of iterations: " * string(solver.current_iteration) * "\n" *
            "Time elapsed: " * string(solver.solve_time))
        @info log_msg
    catch e
        # TODO: exception handling not working
        if e isa LAPACKException
            solver.status_termination = NUMERICAL_ERROR
        end
    end
end

function get_objective(P::AbstractArray{Float64},
                       c::AbstractArray{Float64},
                       x::AbstractArray{Float64})
    return x' * P * x + c' * x
end

function check_preconditions(solver::Solver)
    if rank(solver.program.A) < size(solver.program.A)[1]
        solver.status_termination = INFEASIBLE
        @error "Values of A are inconsistent or redundant."
        # TODO: throw exception?
        @assert false
    end
    n = size(solver.program.P)[1]
    if rank(solver.program.KKT_A[1:n, :]) < n
        solver.status_termination = INFEASIBLE
        @error "There are some constraints in the problem that are either
        redundant or inconsistent."
        @assert false
    end
end

function optimize_main!(solver::Solver)
    # using Conic QP (Primal-Dual Interior Point) method
    # given strictly feasible x, t := t^(0) > 0, mu > 1, tolerance eps > 0.
    primal_obj = solver.obj_primal_value
    total_time_elapsed = 0
    @info "Executing main optimization loop"
    while true
        i = solver.current_iteration
        if i > solver.max_iterations
            # TODO: use duality gap to determine slow progress
            if abs(primal_obj - solver.obj_primal_value) > solver.tol_gap_abs
                solver.status_termination = SLOW_PROGRESS
                break
            end
            solver.status_termination = ITERATION_LIMIT
            break
        end
        itr_time_elapsed = @elapsed begin
        P = solver.program.P
        c = solver.program.c
        x = @view solver.program.KKT_x[1:size(P)[1]]
        # if primal_obj <= solver.limit_obj
        #     solver.status_termination = OBJECTIVE_LIMIT
        # end
        tol = solver.tol_optimality
        result, r, μ = get_solver_status(solver.program, tol)
        if result == true
            solver.status_termination = OPTIMAL
            break
        end
        get_central_path(solver.program, solver.current_iteration, r, μ)
        primal_obj = get_objective(P, c, x)
        solver.obj_primal_value = primal_obj
        log_msg = ("Primal objective value: " * string(solver.obj_primal_value))
        @info log_msg
        end
        solver.current_iteration += 1
        total_time_elapsed = total_time_elapsed + itr_time_elapsed
        solver.solve_time = total_time_elapsed
        if total_time_elapsed > solver.time_limit_sec
            solver.status_termination = TIME_LIMIT
            break
        end
    end
end

function initialize!(solver::Solver)
    if isnothing(solver.limit_obj)
        solver.limit_obj = 0
    end
    if isnothing(solver.limit_soln)
        solver.limit_soln = 0
    end
    if isnothing(solver.max_iterations)
        solver.max_iterations = 5
    end
    if isnothing(solver.num_threads)
        solver.num_threads = 1
    end
    # if isnothing(solver.time_limit_sec)
    solver.time_limit_sec = 1e6
    # end
    solver.obj_dual_value = -1e9
    solver.obj_primal_value = 1e9
    if isnothing(solver.status_termination)
        solver.status_termination = OPTIMIZE_NOT_CALLED
    end
    if isnothing(solver.tol_gap_abs)
        solver.tol_gap_abs = 1e-6
    end
    if isnothing(solver.tol_gap_rel)
        solver.tol_gap_rel = 1e-6
    end
    if isnothing(solver.tol_optimality)
        solver.tol_optimality = 1e-6
    end
    log_solver_parameters(solver)
    initialize!(solver.program)
end

function print_table(data)
    pretty_table(
        data;
        body_hlines        = [1],
        body_hlines_format = Tuple('─' for _ = 1:4),
        cell_alignment     = Dict((1, 1) => :l, (7, 1) => :l),
        formatters         = ft_printf("%10.1f", 2),
        show_header        = false,
        tf                 = tf_borderless,
        highlighters       = (
            hl_cell([(1, 1)], crayon"bold"),
            hl_col(2, crayon"dark_gray"),
            ),
    )
end

function log_solver_parameters(solver::Solver)
    data = ["Solver parameters" "";
        "Objective limit: " solver.limit_obj;
        "Limit solution: " solver.limit_soln;
        "Max iterations: " solver.max_iterations;
        "Num constraints: " get_num_constraints(solver.program);
        "Num threads: " solver.num_threads;
        "Time limit (seconds): " solver.time_limit_sec;
        "Tolerance gap absolute: " solver.tol_gap_abs;
        "Tolerance gap relative: " solver.tol_gap_rel;
        "Tolerance optimality: " solver.tol_optimality;
        ]
    print_table(data)
end

function log_iteration_status(r_x::AbstractArray,
                              r_y::AbstractArray,
                              r_z::AbstractArray,
                              g::Float64,
                              result::Bool)
    data = ["Status" "";
    "Residual x: " sum(r_x);
    "Residual y: " sum(r_y);
    "Residual z: " sum(r_z);
    "Duality gap: " g;
    "Is optimal: " result;]
    print_table(data)
end

function initialize_KKT_A!(program::ConeQP)
    A = program.A
    G = program.G
    P = program.P
    num_rows = size(P)[1] + size(A)[1] + size(G)[1]
    num_cols = size(P)[2] + size(A')[2] + size(G')[2]

    program.KKT_A = zeros((num_rows, num_cols))
    P_row_inds = 1:size(P)[1]
    P_col_inds = 1:size(P)[2]
    program.KKT_A[P_row_inds, P_col_inds] = P
    A_row_inds = P_row_inds[end]+1:P_row_inds[end]+size(A)[1]
    A_col_inds = P_col_inds[end]+1:P_col_inds[end]+size(A')[2]
    program.KKT_A[A_row_inds, P_col_inds] = A
    program.KKT_A[P_col_inds, A_col_inds] = A'
    G_row_inds = A_row_inds[end]+1:A_row_inds[end]+size(G)[1]
    G_col_inds = A_col_inds[end]+1:A_col_inds[end]+size(G')[2]
    program.KKT_A[G_row_inds, P_col_inds] = G
    program.KKT_A[P_col_inds, G_col_inds] = G'
end

function update_cones(program::ConeQP,
                      s=nothing,
                      z=nothing)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]
        if isnothing(s)
            s_vec = @view program.s[inds+1:program.cones_inds[k+1]]
        else
            s_vec = s[inds+1:program.cones_inds[k+1]]
        end
        if isnothing(z)
            z_vec = @view program.z[inds+1:program.cones_inds[k+1]]
        else
            z_vec = z[inds+1:program.cones_inds[k+1]]
        end
        update(cone, s_vec, z_vec)
    end
end

function alpha_p(program::ConeQP)
    α_vec = map(k -> alpha_p(k), program.cones)
    return -α_vec[argmax(abs.(α_vec))]
end

function alpha_d(program::ConeQP)
    α_vec = map(k -> alpha_d(k), program.cones)
    return α_vec[argmax(abs.(α_vec))]
end

function get_num_constraints(program::ConeQP)
    num_constraints = size(program.P)[1] + size(program.A)[1] + size(program.G)[1]
    return num_constraints
end

function initialize!(program::ConeQP)
    G = program.G
    P = program.P

    initialize_KKT_A!(program)
    # save("/home/alexander/Documents/alexander_leong/IPMPSDSolver.jl/dataprep.jld", "prog", program)
    program.KKT_b = vcat([-program.c, program.b, program.h]...)
    program.s = zeros((size(program.G)[1]))
    program.z = zeros((size(program.G)[1]))

    # initialize cones
    e = []
    for (_, cone) in enumerate(program.cones)
        e = vcat(e, get_e(cone))
    end

    # if primal dual starting points not given, initialize as follows
    # see page 13 of coneprog.pdf, Vandenberghe for more details
    inv_W = Matrix{Float64}(I, size(G)[1], size(G)[1])
    G_scaled = -inv_W' * program.G
    b_z = @view program.KKT_b[program.inds_h]
    inv_W_b_z = -inv_W * b_z
    x = qp_solve(program, G_scaled, inv_W_b_z)
    x_inds = 1:size(P)[1]
    program.z = -(program.h - G*x[x_inds])
    update_cones(program)
    α_p = alpha_p(program)
    if α_p < 0
        program.s = -program.z
    else
        program.s = -program.z + α_p * e + e
    end
    α_d = alpha_d(program)
    if α_d >= 0
        program.z = program.z + α_d * e + e
    end
    update_cones(program)
    
    x[program.inds_h] = program.z
    program.KKT_x = x
    
    return 0
end

function check_optimality_conditions(program::ConeQP,
                                     tol::Float32,
                                     r::Vector{Float64},
                                     s::Vector{Float64},
                                     z::AbstractArray)
    x_inds = program.inds_c
    y_inds = program.inds_b
    z_inds = program.inds_h
    b_x = r[x_inds]
    b_y = -r[y_inds]
    b_z = -r[z_inds] - program.s
    b_x_is_optimal = all(x->abs(x) <= tol, b_x)
    b_y_is_optimal = all(x->abs(x) <= tol, b_y)
    b_z_is_optimal = all(x->abs(x) <= tol, b_z)
    x = program.KKT_x
    
    r1 = b_x_is_optimal && b_y_is_optimal && b_z_is_optimal
    r2 = all(s->s >= tol, s) && all(z->z >= tol, z)
    r3 = abs(z' * s) <= tol
    if r1 && r2 && r3
        return true
    end
    return false
end

function get_inv_weighted_mat(program::ConeQP,
                              V::AbstractArray,
                              transpose=false)
    ncols = length(size(V)) == 1 ? 1 : size(V)[2]
    inv_W_V = zeros((size(V)[1], ncols))
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        inv_W_V[inds, :] = get_inv_weighted_mat(cone, V[inds, :], transpose)
    end
    return inv_W_V
end

function get_weighted_mat(program::ConeQP,
                          V::AbstractArray,
                          update_var=false)
    ncols = length(size(V)) == 1 ? 1 : size(V)[2]
    W_V = zeros((size(V)[1], ncols))
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        W_V[inds, :] = get_weighted_mat(cone, V[inds, :])
        if update_var == true
            cone.λ = W_V[inds, 1]
        end
    end
    return W_V
end

function evaluate_residual(qp::ConeQP,
                           Δx::Vector{Float64},
                           x_inds::UnitRange{Int64},
                           y_inds::UnitRange{Int64},
                           z_inds::UnitRange{Int64})
    r = zeros(length(Δx))
    x = Δx[x_inds]
    y = Δx[y_inds]
    z = Δx[z_inds]
    r_x = qp.P * x + qp.A' * y + qp.G' * z + qp.c
    r_y = qp.A * x - qp.b
    r_z = qp.G * x - qp.h
    r[x_inds] = r_x
    r[y_inds] = r_y
    r[z_inds] = r_z
    return r
end

function get_step_size(program::ConeQP,
                       Δs_scaled::AbstractArray{Float64},
                       Δz_scaled::AbstractArray{Float64},
                       α_div=1)
    # see page 12 (step 3. and 5.) and page 23 of coneprog.pdf for implementation
    α = zeros(length(program.cones))
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        αₖ = get_step_size(cone, Δs_scaled[inds], Δz_scaled[inds])
        α[k] = αₖ * α_div
    end
    α = minimum(α)
    # α ϵ [0, 1]
    α = minimum((1, α))
    α = maximum((0, α))
    return α
end

function get_affine_direction(program::ConeQP,
                              x::AbstractArray{Float64})
    x_inds = program.inds_c
    z_inds = program.inds_h
    b = @view program.KKT_b[z_inds]
    s = program.s
    Δx = @view x[x_inds]
    b_z = program.G * Δx - b
    b_z_scaled = get_inv_weighted_mat(program, b_z)
    Δz = get_inv_weighted_mat(program, b_z_scaled, true)
    Δs = -s - b_z
    x[z_inds] = Δz[:, 1]
    Δz_scaled = b_z_scaled
    Δs_scaled = get_inv_weighted_mat(program, Δs)
    @info "Calculated affine direction"
    return Δs, Δs_scaled, Δz_scaled, x
end

function get_combined_direction(program::ConeQP,
                                d_z::AbstractArray{Float64},
                                x::AbstractArray{Float64})
    G = program.G
    x_inds = program.inds_c
    z_inds = program.inds_h
    b_z = @view program.KKT_b[z_inds]
    b = b_z - G * x[x_inds]
    W_b = get_inv_weighted_mat(program, b)
    Δz = -get_inv_weighted_mat(program, W_b, true)
    Δs = d_z - b_z + b
    x[z_inds] = Δz[:, 1]
    Δz_scaled = get_weighted_mat(program, Δz[:, 1])
    Δs_scaled = get_inv_weighted_mat(program, Δs)
    @info "Calculated combined direction"
    return Δs, Δs_scaled, Δz_scaled, x
end

function get_solver_status(program::ConeQP,
                           tol::Float32)
    x = program.KKT_x
    s = program.s

    # range objects for indexing
    x_inds = program.inds_c
    y_inds = program.inds_b
    z_inds = program.inds_h
    
    # evaluate residual
    r = evaluate_residual(program, x, x_inds, y_inds, z_inds)
    r_x = @view r[x_inds]
    r_y = @view r[y_inds]
    r_z = @view r[z_inds]
    
    # evaluate gap
    m = 0
    for cone in program.cones
        m = m + degree(cone)
    end
    μ = s' * x[z_inds] / m

    # evaluate stopping criteria
    z = @view x[z_inds]
    result = check_optimality_conditions(program, tol, r, s, z)
    log_iteration_status(r_x, r_y, r_z, μ, result)

    return result, r, μ
end

function update_iterates(program::ConeQP,
                         s::AbstractArray{Float64},
                         Δs::AbstractArray{Float64},
                         Δx::AbstractArray{Float64})
    s = s + program.α * Δs
    Δx = program.KKT_x + program.α * Δx
    program.KKT_x = Δx
    program.s = s[:, 1]
    z_inds = program.inds_h
    program.z = Δx[z_inds]
    @info "Updated iterates"
end

function get_central_path(program::ConeQP,
                          current_itr::Int32,
                          r::AbstractArray{Float64},
                          μ::Float64)
    # range objects for indexing
    z_inds = program.inds_h

    # Affine direction, i.e. solve Newton equations
    # see page 29 of coneprog.pdf for solving a KKT system with SDP constraints
    # get scaling factors
    λ = zeros(length(program.s))
    if current_itr == 1
        # W defined as R on page 10, coneprog.pdf
        for (k, cone) in enumerate(program.cones)
            inds = program.cones_inds[k]+1:program.cones_inds[k+1]
            Wₖ, inv_Wₖ, λₖ = get_scaling_vars(cone)
            cone.W = Wₖ
            cone.inv_W = inv_Wₖ
            λ[inds] = λₖ
        end
    else
        for (k, cone) in enumerate(program.cones)
            inds = program.cones_inds[k]+1:program.cones_inds[k+1]
            λ[inds] = cone.λ
        end
    end
    # TODO: compute weighted program.G here!

    # solve linear equations to get affine direction
    # see page 14 of coneprog.pdf for setting z part of b
    s = program.s
    program.KKT_b = -r
    program.KKT_b[z_inds] = -r[z_inds] + s
    G_scaled = get_inv_weighted_mat(program, program.G)
    b_z = @view program.KKT_b[program.inds_h]
    inv_W_b_z = get_inv_weighted_mat(program, b_z)
    Δx = qp_solve(program, G_scaled, inv_W_b_z)
    Δsₐ, Δsₐ_scaled, Δzₐ_scaled, Δxₐ = get_affine_direction(program, Δx)
    update_cones(program, Δsₐ, Δxₐ[z_inds])
    
    @assert check_affine_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled)

    # Compute step size and centering parameter
    α = get_step_size(program, Δsₐ_scaled, Δzₐ_scaled)
    @assert is_convex_cone(program, α)
    ρ = 1 - α + α^2 * dot(Δsₐ_scaled', Δzₐ_scaled) / dot(λ', λ)
    σ = maximum((0, minimum((1, ρ))))^3
    @info "Step size: " α
    @info "Centering parameter: " σ
    
    # Combined direction, i.e. solve linear equations
    # see page 29 of coneprog.pdf for solving a KKT system with SDP constraints
    γ = 1 # TODO: expose Mehrotra correction parameter
    η = 0
    KKT_b = -(1 - η) * r
    b_z = copy(KKT_b[z_inds])
    # see eq. 19a, 19b, 22a of coneprog.pdf
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        b_z_k = b_z[inds]
        b_z_k = get_d_s(cone, Δsₐ_scaled[inds], Δzₐ_scaled[inds], b_z_k, γ, λ[inds], μ, σ)
        KKT_b_z = @view program.KKT_b[z_inds]
        KKT_b_z[inds] = b_z_k
    end
    Δx = qp_solve(program, G_scaled, inv_W_b_z)
    Δs, Δs_scaled, Δz_scaled, Δx = get_combined_direction(program, KKT_b[z_inds], Δx)
    Δz = @view Δx[z_inds]
    update_cones(program, Δs, Δz)

    @assert check_combined_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled, Δs_scaled, Δz_scaled, μ, σ)
    
    # update step size
    program.α = get_step_size(program, Δs_scaled, Δz_scaled, 0.99)
    @assert is_convex_cone(program, program.α)
    @info "Updated Step size: " * string(program.α)

    # update iterates with updated α
    update_iterates(program, s, Δs, Δx)

    # update scaling matrices and vars
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        update_scaling_vars(cone, Δs_scaled[inds], Δz_scaled[inds], program.α)
    end
    @info "Updated scaling matrices and scaling vars"
end