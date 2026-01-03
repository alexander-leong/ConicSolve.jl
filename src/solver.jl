#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""see CVXOPT Vandenberghe 2010 for Cone QP definition"""

# include("./arrayutils.jl")
include("./debug.jl")
include("./kktsolver.jl")
include("./program.jl")

using DataStructures
using LinearAlgebra
using Logging
using SparseArrays

@enum ResultStatus begin
    NO_SOLUTION
    FEASIBLE_POINT
    INFEASIBLE_POINT
end

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

@enum Device begin
    CPU
    GPU
end

export Device, CPU, GPU

function Base.parse(::Type{Device}, value)
    return value == "CPU" ? CPU : GPU
end

export parse

mutable struct SolverStatus
    current_iteration::Int32
    kkt_iterate::CircularBuffer{KKTIterate}

    duality_gap::AbstractArray{Float64}
    dual_obj::AbstractArray{Float64}
    primal_obj::AbstractArray{Float64}
    residual_x::AbstractArray{Float64}
    residual_y::AbstractArray{Float64}
    residual_z::AbstractArray{Float64}
    step_size::AbstractArray{Float64}
    status_termination::Union{TerminationStatus, Nothing}

    function SolverStatus()
        status = new()
        status.current_iteration = 0
        buffer_size = 5
        status.kkt_iterate = CircularBuffer{KKTIterate}(buffer_size)
        status.duality_gap = []
        status.dual_obj = []
        status.primal_obj = []
        status.residual_x = []
        status.residual_y = []
        status.residual_z = []
        status.step_size = []
        status.status_termination = OPTIMIZE_NOT_CALLED
        return status
    end
end

function get_best_iterate(status::SolverStatus)
    rx_weight = 1
    ry_weight = 1
    rz_weight = 1
    gap_weight = 1
    solution_scores = []
    for (i, _) in enumerate(status.kkt_iterate)
        offset = length(status.kkt_iterate) - i
        push!(solution_scores,
        rx_weight * status.residual_x[end-offset] +
        ry_weight * status.residual_y[end-offset] +
        rz_weight * status.residual_z[end-offset] +
        gap_weight * status.duality_gap[end-offset])
    end
    best_idx = argmin(solution_scores)
    i = status.current_iteration - best_idx + 1
    @info "The primal/dual variables correspond to iterate $(i)"
    return status.kkt_iterate[best_idx]
end

"""
    Solver

Represents an Interior Point Method solver
for solving Conic Quadratic Programs.
All parameters are optional except `program`.

### Parameters:
* `cb_after_iteration`: Callback function of the form: function cb(solver::Solver) end
* `cb_before_iteration`: Callback function of the form: function cb(solver::Solver) end
* `device`: CPU or GPU
* `program`: The Cone QP to solve
* `kktsolver`: The object to represent the KKT solver used to solve the KKT system. The `kktsolve` argument will construct the appropriate KKTSolver object. \
Possible values for kktsolve are `conjgrad`, `minres` and `qrchol`. `qrchol` is the default.
* `limit_obj`: The minimum/maximum objective value the solver will terminate
* `limit_soln`: The 2-norm difference between the current and previous estimates
* `max_iterations`: The maximum number of iterations before the solver terminates
* `num_threads`: The number of threads used on the CPU to perform certain BLAS and parallelized operations
* `time_limit_sec`: The maximum number of seconds elapsed before the solver terminates after the current iteration
* `tol_gap_abs`: The absolute gap tolerance
* `tol_gap_rel`: The relative gap tolerance
* `tol_optimality`: The absolute tolerance for satisfying the optimality conditions
* `η`: Optimization parameter typically set to zero or σ, default is 0.0. Set η as nothing to set to σ.
* `γ`: Mehrotra correction parameter set to γ ∈ [0, 1], default is 1.0
"""
mutable struct Solver
    cb_after_iteration
    cb_before_iteration
    current_iteration::Int32
    device::Device
    kktsolver::KKTSolver
    limit_obj::Union{Float64, Nothing}
    limit_soln::Union{Float64, Nothing}
    log_level
    max_iterations::Union{Int8, Nothing}
    num_threads::Union{Int8, Nothing}
    obj_dual_value::Float64
    obj_primal_value::Float64
    program::ConeQP
    solve_time
    status::SolverStatus
    status_dual::ResultStatus
    status_primal::ResultStatus
    status_termination::Union{TerminationStatus, Nothing}
    time_limit_sec::Int32
    tol_gap_abs::Union{Float64, Nothing}
    tol_gap_rel::Union{Float64, Nothing}
    tol_optimality::Union{Float64, Nothing}
    η
    γ::Float64 # Mehrotra correction parameter

    """
        Solver(program, kktsolve, η, γ)
    
    Constructs a new QP solver object.
    """
    function Solver(program::Union{ConeQP, Missing}=missing,
                    kktsolve="qrchol",
                    preconditioner="none",
                    limit_obj=-Inf,
                    limit_soln=0,
                    tol_gap_abs=1e-3,
                    tol_gap_rel=1e-3,
                    tol_optimality=1e-3,
                    max_iterations=100,
                    time_limit_sec=1e6,
                    η=0.0,
                    γ::Float64=1.0,
                    cb_before_iteration=nothing,
                    cb_after_iteration=nothing)
        solver = new()
        solver.cb_after_iteration = cb_after_iteration
        solver.cb_before_iteration = cb_before_iteration
        solver.current_iteration = 1
        solver.device = CPU
        solver.kktsolver = setup_default_kkt_solver(kktsolve, preconditioner)
        if !ismissing(program)
            check_program(program)
            solver.program = program
        end
        solver.limit_obj = limit_obj
        solver.limit_soln = limit_soln
        solver.max_iterations = max_iterations
        solver.num_threads = 1
        solver.solve_time = 0
        solver.status = SolverStatus()
        solver.status_dual = NO_SOLUTION
        solver.status_primal = NO_SOLUTION
        solver.time_limit_sec = time_limit_sec
        solver.tol_gap_abs = tol_gap_abs
        solver.tol_gap_rel = tol_gap_rel
        solver.tol_optimality = tol_optimality
        solver.η = η
        solver.γ = γ
        return solver
    end
end

"""
    get_solver_status(solver)

Returns the object of type `SolverStatus` used by the solver.
"""
function get_solver_status(solver::Solver)
    return solver.status
end

function update_dual_status(solver::Solver)
    status = solver.status
    dual_res = status.residual_x[end]
    tol = solver.tol_optimality
    if dual_res <= tol
        solver.status_dual = FEASIBLE_POINT
    else
        solver.status_dual = INFEASIBLE_POINT
    end
    return dual_res <= tol
end

function update_primal_status(solver::Solver)
    status = solver.status
    primal_res = status.residual_y[end]
    tol = solver.tol_optimality
    if primal_res <= tol
        solver.status_primal = FEASIBLE_POINT
    else
        solver.status_primal = INFEASIBLE_POINT
    end
    return primal_res <= tol
end

function within_tol(abs_tol, rel_tol, value)
    if value / abs_tol < rel_tol
        return true
    end
    return false
end

"""
    get_solution(solver)

Get the solution to the optimization problem.
"""
function get_solution(solver::Solver)
    x_inds = solver.program.inds_c
    return solver.program.KKT_x[x_inds]
end

function print_graphic()
    fp = open("graphic.txt", "r")
    words = readlines(fp, keep=true)
    for word in words
        print(word)
    end
end

function print_header()
    # print_graphic()
    println("Alexander Leong, 2025")
    println("Version 0.0.3")
    println(repeat("-", 144))
end

"""
    run_solver(solver)

Executes the solver on the optimization problem.
"""
function run_solver(solver::Solver, is_init=false, check=false, kwargs...)
    try
        print_header()
        @info "Optimize called"
        BLAS.set_num_threads(solver.num_threads)
        # if !isnothing(solver.cb_before_iteration)
            # solver.cb_before_iteration(kwargs...)
        # end

        # initialize cones
        program = solver.program
        if program.cones_inds == []
            set_cones_inds(program)
        end

        initialize!(solver, is_init)
        if check == true
            check_preconditions(solver)
        end
        result = optimize_main!(solver, kwargs...)
        return result
    catch err
        if isa(err, LAPACKException)
            solver.status.status_termination = NUMERICAL_ERROR
        end
    finally
        log_msg = ("Solver finished" * "\n" *
            "Exit status: " * string(solver.status.status_termination) * "\n" *
            "Primal objective value: " * string(solver.obj_primal_value) * "\n" *
            "Number of iterations: " * string(solver.current_iteration - 1) * "\n" *
            "Time elapsed: " * string(solver.solve_time))
        @info log_msg
        return solver.status
    end
end

"""
    get_objective(P, c, x)

Get the current objective value to the conic quadratic program
`` x^TPx + c^Tx ``
"""
function get_objective(P::Union{AbstractArray{Float64}, Nothing},
                       c::AbstractArray{U},
                       x::AbstractArray{T}) where {T<:Number, U<:Number}
    obj = 0.0
    if !isnothing(P)
        obj = x' * P * x
    end
    obj = obj + c' * x
    return obj
end

function check_preconditions(solver::Solver)
    qp = solver.program
    if !isnothing(qp.A) && rank(qp.A) < size(qp.A)[1]
        solver.status.status_termination = INFEASIBLE
        @error "Values of A are inconsistent or redundant."
        @assert false
    end
    if !isnothing(qp.A)
        if rank([qp.P qp.A' qp.G']) < size(qp.P)[1]
            solver.status.status_termination = INFEASIBLE
            @error "There are some constraints in the problem that are either
            redundant or inconsistent."
            @assert false
        end
    else
        if rank([qp.P qp.G']) < size(qp.P)[1]
            solver.status.status_termination = INFEASIBLE
            @error "There are some constraints in the problem that are either
            redundant or inconsistent."
            @assert false
        end
    end
end

function optimize_main!(solver::Solver, kwargs...)
    # using Conic QP (Primal-Dual Interior Point) method
    # given strictly feasible x, t := t^(0) > 0, mu > 1, tolerance eps > 0.
    solver.obj_primal_value = Inf
    total_time_elapsed = 0
    @info "Executing main optimization loop"
    while true
        # if !isnothing(solver.cb_before_iteration)
        #     solver.cb_before_iteration(kwargs...)
        # end
        i = solver.current_iteration
        itr_time_elapsed = @elapsed begin
        result, r, μ = update_solver_status(solver)
        if result == true
            @info "Terminating optimization loop"
            break
        end
        η = solver.η
        γ = solver.γ
        get_central_path(solver, i, r, μ, η, γ)
        end

        solver.current_iteration += 1
        if i >= solver.max_iterations
            solver.status.status_termination = ITERATION_LIMIT
            break
        end

        if !isnothing(solver.cb_after_iteration)
            solver.cb_after_iteration(kwargs...)
        end

        total_time_elapsed = total_time_elapsed + itr_time_elapsed
        solver.solve_time = total_time_elapsed
        if total_time_elapsed > solver.time_limit_sec
            solver.status.status_termination = TIME_LIMIT
            break
        end
    end

    # update program with best in-memory solution
    kkt_iterate = get_best_iterate(solver.status)
    program.KKT_x = kkt_iterate.KKT_x
    program.s = kkt_iterate.s
    program.z = kkt_iterate.z
    return status
end

function print_table(data, i, n=100, pad=15)
    if i % n == 1
        row = "| "
        for val in eachrow(data)
            row *= lpad(val[:][1], pad, " ") * " |"
        end
        println(row)
    end
    row = "| "
    for val in eachrow(data)
        row *= lpad(round(val[:][end], sigdigits=6), pad, " ") * " |"
    end
    println(row)
end

function log_solver_parameters(solver::Solver)
    pad = 32
    data = [lpad("Solver parameters", pad, " ") "";
        lpad("Objective limit: ", pad, " ") solver.limit_obj;
        lpad("Limit solution: ", pad, " ") solver.limit_soln;
        lpad("Max iterations: ", pad, " ") solver.max_iterations;
        lpad("Num constraints: ", pad, " ") get_num_constraints(solver.program);
        lpad("Num threads: ", pad, " ") solver.num_threads;
        lpad("Preferred device: ", pad, " ") solver.device;
        lpad("System Solver KKT method: ", pad, " ") solver.kktsolver.kktsolve["label"];
        lpad("Time limit (seconds): ", pad, " ") solver.time_limit_sec;
        lpad("Tolerance gap absolute: ", pad, " ") solver.tol_gap_abs;
        lpad("Tolerance gap relative: ", pad, " ") solver.tol_gap_rel;
        lpad("Tolerance optimality: ", pad, " ") solver.tol_optimality;
        ]
    for item in eachrow(data)
        println("$(item[1]) $(item[2])")
    end
end

function log_iteration_status(i::Int32,
                              r_x::AbstractArray,
                              r_y::AbstractArray,
                              r_z::AbstractArray,
                              g::T,
                              pri_obj,
                              dual_obj,
                              α) where T <: Number
    data = ["Iter." i;
    "Dual Res." norm(r_x);
    "Primal Res." norm(r_y);
    "Cent. Res." norm(r_z);
    "Dual. Gap" norm(g);
    "Primal Obj." pri_obj;
    "Dual Obj." dual_obj;
    "Step size" α;]
    # TODO logging frequency and other options
    n = 100
    print_table(data, i, n)
end

function update_cones(program::ConeQP,
                      s=nothing,
                      z=nothing)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]
        if isnothing(s)
            s_vec = program.s[inds+1:program.cones_inds[k+1]]
        else
            s_vec = s[inds+1:program.cones_inds[k+1]]
        end
        if isnothing(z)
            z_vec = program.z[inds+1:program.cones_inds[k+1]]
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
    return -minimum(α_vec)
end

function get_num_constraints(program::ConeQP)
    num_constraints = 0
    if !isnothing(program.A)
        num_constraints += size(program.A)[1]
    end
    num_constraints += size(program.G)[1]
    return num_constraints
end

function initialize_solver!(solver::Solver)
    program = solver.program
    G = @view program.G[:, :]
    program.s = zeros((size(G)[1]))
    program.z = zeros((size(G)[1]))
    e = []
    for (_, cone) in enumerate(program.cones)
        e = vcat(e, get_e(cone))
    end

    @info "Solving for primal dual starting points"
    b_z = @view program.KKT_b[program.inds_h]
    inv_W_b_z = -b_z
    kkt_1_1 = G' * G
    if !isnothing(program.P)
        kkt_1_1 = program.P + kkt_1_1
    end
    kktsystem = program.kktsystem
    kktsystem.kkt_1_1 = kkt_1_1
    x_hat = qp_solve(solver, G, inv_W_b_z, qr_chol_solve)
    x_inds = 1:size(G)[2]
    program.z = -(program.h - G*x_hat[x_inds])
    z_hat = program.z
    update_cones(program)
    α_p = alpha_p(program)
    if α_p < 0
        program.s = -z_hat
    else
        program.s = -z_hat + α_p * e + e
    end
    α_d = alpha_d(program)
    if α_d >= 0
        program.z = z_hat + α_d * e + e
    end
    x_hat[program.inds_h] = program.z
    program.KKT_x = x_hat
    update_cones(program)
    
    return 0
end

function update_affine_constraints(program::ConeQP)
    program.kktsystem.A = program.A
    program.kktsystem.b = program.b
end

function update_inequality_constraints(program::ConeQP)
    program.kktsystem.G = program.G
    program.kktsystem.h = program.h
end

export update_affine_constraints
export update_inequality_constraints

function warm_start!(solver::Solver)
    program = solver.program
    update_cones(program)
end

function initialize!(solver::Solver, is_init=false)
    solver.status.status_termination = INFEASIBLE
    log_solver_parameters(solver)
    program = solver.program
    A = program.A
    G = program.G
    P = program.P

    program.kktsystem = !isnothing(program.A) ? KKTSystem(A, G, P) : KKTSystem(G, P)
    if !isnothing(program.b)
        program.KKT_b = vcat([-program.c, program.b, program.h]...)
    else
        program.KKT_b = vcat([-program.c, program.h]...)
    end

    if is_init
        warm_start!(solver)
    else
        initialize_solver!(solver)
    end
end

function check_duality_gap(μ,
                           atol::Float64=1e-3,
                           rtol::Float64=1e-3)
    return within_tol(atol, rtol, μ)
end

function check_feasibility(solver::Solver,
                           tol::Float64)
    is_feasible_status = solver.status_dual == FEASIBLE_POINT &&
                         solver.status_primal == FEASIBLE_POINT
    is_feasible = is_feasible_status &&
                  is_convex_cone(solver.program, 0, tol)
    return is_feasible
end

function check_infeasibility(solver::Solver,
                             tol::Float64=1e-3)
    program = solver.program
    y_inds = program.inds_b
    A = program.A
    b = program.b
    y = program.KKT_x[y_inds]
    if norm(y) > 1/tol
        inds = [get_indices_of_constraint(program, cone) for cone in program.cones]
        y_is_relint = [is_convex_cone(cone, A[:, inds[i]]' * y) for (i, cone) in enumerate(program.cones)]
        y_in_cone = all(y_is_relint) || true
        return y_in_cone && b' * y - tol < 0
    end
end

function check_progress(solver::Solver, tol::Float64)
    status = solver.status
    if length(status.step_size) > 0 &&
    status.step_size[end] < tol
        if length(status.duality_gap) > 1 &&
        abs(status.duality_gap[end] - status.duality_gap[end-1]) < tol
            @info "Slow progress..."
            status.status_termination = SLOW_PROGRESS
        end
    end
end

"""
    is_optimal(solver, gap_atol, gap_rtol, tol)

The stopping criterion used to determine convergence.
Convergence is based on the following criteria:
- residuals close to zero (within tol)
- solution is primal-dual feasible
- duality gap close to zero (within tol)
- minimal change in primal objective value
"""
function is_optimal(solver::Solver,
                    μ,
                    gap_atol::Float64,
                    gap_rtol::Float64,
                    tol::Float64)
    r1 = check_feasibility(solver, tol)

    r2 = check_duality_gap(μ, gap_atol, gap_rtol)

    # stopping criteria based on
    # https://www.seas.ucla.edu/~vandenbe/ee236a/lectures/mpc.pdf
    if r1 && r2 == true
        status.status_termination = OPTIMAL
        return true
    end

    status = solver.status
    if abs(solver.obj_primal_value) <= solver.limit_obj
        status.status_termination = OBJECTIVE_LIMIT
        return true
    end
    i = solver.current_iteration
    if i == 0
        return false
    end

    # check if convergence has stalled
    check_progress(solver, tol)
    return false
end

function get_inv_weighted_mat(program::ConeQP,
                              V::AbstractArray{T},
                              transpose=false) where T <: Number
    ncols = length(size(V)) == 1 ? 1 : size(V)[2]
    inv_W_V = zeros(T, (size(V)[1], ncols))
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        Vi = @view V[inds, :]
        inv_W_V[inds, :] = get_inv_weighted_mat(cone, Vi, transpose)
    end
    return inv_W_V
end

function get_weighted_mat(program::ConeQP,
                          V::AbstractArray{T},
                          update_var=false) where T <: Number
    ncols = length(size(V)) == 1 ? 1 : size(V)[2]
    W_V = zeros(T, (size(V)[1], ncols))
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        Vi = @view V[inds, :]
        W_V[inds, :] = get_weighted_mat(cone, Vi)
        if update_var == true
            cone.λ = W_V[inds, 1]
        end
    end
    return W_V
end

function get_dual_value(qp::ConeQP,
                        Δx::Vector{Float64},
                        y_inds::UnitRange{Int64},
                        z_inds::UnitRange{Int64})
    z = @view Δx[z_inds]
    v = 0
    Pw = qp.G' * z + qp.c
    if !isnothing(qp.A)
        y = @view Δx[y_inds]
        Pw = Pw + (qp.A' * y)
        v = v + (-qp.b' * y)
    end
    if all(qp.P .== 0) == false
        w = qp.inv_P * Pw
        v = v + (-(1/2) * w' * qp.P * w)
    end
    v += -qp.h' * qp.z
    return Pw, v
end

function evaluate_residual(qp::ConeQP,
                           Δx::Vector{T},
                           Pw::Vector{Float64},
                           x_inds::UnitRange{Int64},
                           y_inds::UnitRange{Int64},
                           z_inds::UnitRange{Int64}) where T <: Number
    r = zeros(T, length(Δx))
    x = @view Δx[x_inds]
    z = @view Δx[z_inds]
    P_x = zeros(Float64, length(x))
    if !isnothing(qp.P)
        P_x = qp.P * x
    end
    r_x = P_x + Pw
    if !isnothing(qp.A)
        r_y = qp.A * x - qp.b
        r[y_inds] = r_y
    end
    r_z = qp.s + qp.G * x - qp.h
    r[x_inds] = r_x
    r[z_inds] = r_z
    return r
end

function get_step_size(program::ConeQP,
                       Δs_scaled::AbstractArray{T},
                       Δz_scaled::AbstractArray{T},
                       α_div=1) where T <: Number
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
                              x::AbstractArray)
    x_inds = program.inds_c
    z_inds = program.inds_h
    b = @view program.KKT_b[z_inds]
    s = @view program.s[:]
    Δx = @view x[x_inds]
    b_z = program.G * Δx - b
    b_z_scaled = get_inv_weighted_mat(program, b_z)
    Δz = get_inv_weighted_mat(program, b_z_scaled, true)
    Δs = -s - b_z
    x[z_inds] = Δz[:, 1]
    Δz_scaled = @view b_z_scaled[:]
    Δs_scaled = get_inv_weighted_mat(program, Δs)
    @debug "Calculated affine direction"
    return Δs, Δs_scaled, Δz_scaled, x
end

function get_combined_direction(program::ConeQP,
                                d_z::AbstractArray,
                                x::AbstractArray)
    G = @view program.G[:, :]
    x_inds = program.inds_c
    z_inds = program.inds_h
    b_z = @view program.KKT_b[z_inds]
    G_x = G * x[x_inds]
    b = b_z - G_x
    W_b = get_inv_weighted_mat(program, b)
    Δz = -get_inv_weighted_mat(program, W_b, true)
    Δs = d_z - G_x
    x[z_inds] = Δz[:, 1]
    Δz_scaled = get_weighted_mat(program, Δz[:, 1])
    Δs_scaled = get_inv_weighted_mat(program, Δs)
    @debug "Calculated combined direction"
    return Δs, Δs_scaled, Δz_scaled, x
end

function get_affine_search_direction(solver::Solver,
                                     G_scaled,
                                     kkt_1_1,
                                     inv_W_b_z)
    program = solver.program
    kktsolver_fn = solver.kktsolver.kktsolve["fn"]
    Δx = qp_solve(solver, G_scaled, inv_W_b_z, kktsolver_fn)
    Δsₐ, Δsₐ_scaled, Δzₐ_scaled, Δxₐ = get_affine_direction(program, Δx)
    return Δsₐ, Δsₐ_scaled, Δzₐ_scaled, Δxₐ
end

function get_combined_search_direction(solver::Solver,
                                       G_scaled,
                                       b_z,
                                       kkt_1_1,
                                       inv_W_b_z)
    program = solver.program
    kktsolver_fn = solver.kktsolver.kktsolve["fn"]
    Δx = qp_solve(solver, G_scaled, inv_W_b_z, kktsolver_fn)
    Δs, Δs_scaled, Δz_scaled, Δx = get_combined_direction(program, b_z, Δx)
    return Δs, Δs_scaled, Δz_scaled, Δx
end

function get_iterative_affine_search_direction(solver::Solver,
                                               G_scaled,
                                               kkt_1_1,
                                               inv_W_b_z)
    program = solver.program
    kktsolver_fn = solver.kktsolver.kktsolve["fn"]
    Δx = qp_solve_iterative(solver, G_scaled, kkt_1_1, inv_W_b_z, kktsolver_fn)
    Δx = vcat(Δx, zeros(length(program.inds_h)))
    Δs, Δs_scaled, Δz_scaled, Δx = get_affine_direction(program, Δx)
    return Δs, Δs_scaled, Δz_scaled, Δx
end

function get_iterative_combined_search_direction(solver::Solver,
                                                 G_scaled,
                                                 b_z,
                                                 kkt_1_1,
                                                 inv_W_b_z)
    program = solver.program
    kktsolver_fn = solver.kktsolver.kktsolve["fn"]
    Δx = qp_solve_iterative(solver, G_scaled, kkt_1_1, inv_W_b_z, kktsolver_fn)
    Δx = vcat(Δx, zeros(length(program.inds_h)))
    Δs, Δs_scaled, Δz_scaled, Δx = get_combined_direction(program, b_z, Δx)
    return Δs, Δs_scaled, Δz_scaled, Δx
end

function evaluate_optimality_conditions(solver::Solver)
    i = solver.current_iteration
    program = solver.program
    x = program.KKT_x
    s = @view program.s[:]

    # range objects for indexing
    x_inds = program.inds_c
    y_inds = program.inds_b
    z_inds = program.inds_h

    # get dual values
    Pw, dual_obj = get_dual_value(program, x, y_inds, z_inds)
    
    # evaluate residual
    r = evaluate_residual(program, x, Pw, x_inds, y_inds, z_inds)
    r_x = @view r[x_inds]
    if !isnothing(program.A)
        r_y = @view r[y_inds]
    else
        r_y = [0]
    end
    r_z = @view r[z_inds]
    
    # evaluate gap
    m = 0
    for cone in program.cones
        m = m + degree(cone)
    end
    μ = s' * x[z_inds] / m
    
    pri_obj = solver.obj_primal_value
    
    gap_atol = solver.tol_gap_abs
    gap_rtol = solver.tol_gap_rel
    tol = solver.tol_optimality

    result = is_optimal(solver, μ, gap_atol, gap_rtol, tol)

    # save status
    status = solver.status
    status.current_iteration = i
    push!(status.duality_gap, μ)
    push!(status.dual_obj, dual_obj)
    push!(status.primal_obj, pri_obj)
    push!(status.residual_x, norm(r_x))
    push!(status.residual_y, norm(r_y))
    push!(status.residual_z, norm(r_z))
    push!(status.step_size, program.α)
    dual_res = update_dual_status(solver)
    primal_res = update_primal_status(solver)
    if dual_res && primal_res
        status.status_termination = ALMOST_OPTIMAL
    end

    # TODO only log if making progress
    log_iteration_status(i, r_x, r_y, r_z, μ, pri_obj, dual_obj, program.α)

    return result, r, μ
end

function update_solver_status(solver::Solver)
    # update primal obj.
    program = solver.program
    P = program.P
    c = @view program.c[:]
    x = @view program.KKT_x[1:length(c)]
    primal_obj = get_objective(P, c, x)
    solver.obj_primal_value = primal_obj

    # evaluate stopping criteria
    status = solver.status
    i = solver.current_iteration
    if i > 0
        # Farkas' lemma
        r1 = check_infeasibility(solver)
        if r1 == true
            status.status_termination = INFEASIBLE
            return true, [], nothing
        end
    end
    result, r, μ = evaluate_optimality_conditions(solver)

    return result, r, μ
end

function update_iterates(program::ConeQP,
                         Δs::AbstractArray,
                         Δx::AbstractArray)
    s = program.s + program.α * Δs
    program.s = s[:, 1]
    z_inds = program.inds_h
    Δx = program.KKT_x + program.α * Δx
    program.KKT_x = Δx
    program.z = Δx[z_inds]
    @debug "Updated iterates"
end

function run_on_device(device, fn, args...)
    for arg in args
        arg = get_array(device, arg)
    end
    return fn(args...)
end

function get_central_path(solver::Solver,
                          current_itr::Int32,
                          r::AbstractArray{T},
                          μ::T,
                          η=0.0,
                          γ::Float64=1.0) where T <: Number
    kktsolver = solver.kktsolver
    program = solver.program
    z_inds = program.inds_h

    # get scaling factors
    # get_scaling_factors_elapsed_time = @elapsed begin
    λ = zeros(length(program.s))
    if current_itr == 1
        for (k, cone) in enumerate(program.cones)
            inds = program.cones_inds[k]+1:program.cones_inds[k+1]
            cone.W, cone.inv_W, λ[inds] = get_scaling_vars(cone)
        end
    else
        for (k, cone) in enumerate(program.cones)
            inds = program.cones_inds[k]+1:program.cones_inds[k+1]
            λ[inds] = cone.λ
        end
    end
    # end
    # println(get_scaling_factors_elapsed_time)
    # solve linear equations to get affine direction
    # see page 14 of coneprog.pdf for setting z part of b
    # get_affine_direction_elapsed_time = @elapsed begin
    s = @view program.s[:]
    program.KKT_b = -r
    program.KKT_b[z_inds] = -r[z_inds] + s
    G_scaled = get_inv_weighted_mat(program, program.G)
    b_z = @view program.KKT_b[program.inds_h]
    inv_W_b_z = get_inv_weighted_mat(program, b_z)
    inv_W_b_z = @view inv_W_b_z[:, 1]
    kktsystem = program.kktsystem
    function qr_dev(G_scaled)
        G_scaled_QR_R = qr(G_scaled).R
        gram_G_scaled = G_scaled_QR_R' * G_scaled_QR_R
        kktsystem.kkt_1_1 = zeros(Float32, size(gram_G_scaled))
        copyto!(kktsystem.kkt_1_1, gram_G_scaled)
    end
    run_on_device(solver.device, qr_dev, G_scaled)
    if !isnothing(program.P)
        kktsystem.kkt_1_1 = program.P + kktsystem.kkt_1_1
    end
    Δsₐ, Δsₐ_scaled, Δzₐ_scaled, Δxₐ = kktsolver.affine_search_direction(solver, G_scaled, kktsystem.kkt_1_1, inv_W_b_z)
    update_cones(program, Δsₐ, Δxₐ[z_inds])
    # end
    # println(get_affine_direction_elapsed_time)

    @debug "Affine direction ok?: " check_affine_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled)

    # Compute step size and centering parameter
    # get_step_size_elapsed_time = @elapsed begin
    α = get_step_size(program, Δsₐ_scaled, Δzₐ_scaled)
    @debug "Solution after step ok?: " is_convex_cone(program, α)
    ρ = 1 - α + α^2 * dot(Δsₐ_scaled', Δzₐ_scaled) / dot(λ', λ)
    σ = maximum((0, minimum((1, abs(ρ)))))^3
    if η === nothing
        η = σ
    end
    # end
    # println(get_step_size_elapsed_time)
    @debug "Step size: " α
    @debug "η: " η
    @debug "Centering parameter: " σ
    
    # Combined direction, i.e. solve linear equations
    # see page 29 of coneprog.pdf for solving a KKT system with SOCP and SDP constraints
    # get_combined_direction_elapsed_time = @elapsed begin
    KKT_b = -(1 - η) * r
    program.KKT_b = KKT_b
    b_z = KKT_b[z_inds]
    # see eq. 19a, 19b, 22a of coneprog.pdf
    KKT_b_z = @view program.KKT_b[z_inds]
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        b_z_k = @view b_z[inds]
        Δsₐ_scaled_k = @view Δsₐ_scaled[inds]
        Δzₐ_scaled_k = @view Δzₐ_scaled[inds]
        d_s = get_d_s(cone, Δsₐ_scaled_k, Δzₐ_scaled_k, b_z_k, γ, λ[inds], μ, σ)
        KKT_b_z[inds] = d_s
    end
    inv_W_b_z = get_inv_weighted_mat(program, KKT_b_z)
    inv_W_b_z = @view inv_W_b_z[:, 1]
    Δs, Δs_scaled, Δz_scaled, Δx = kktsolver.combined_search_direction(solver, G_scaled, b_z, kktsystem.kkt_1_1, inv_W_b_z)
    Δx_z = @view Δx[z_inds]
    update_cones(program, Δs, Δx_z)
    # end
    # println(get_combined_direction_elapsed_time)

    @debug "Combined direction ok?:" check_combined_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled, Δs_scaled, Δz_scaled, μ, σ)
    
    # update_elapsed_time = @elapsed begin
    # update step size
    program.α = get_step_size(program, Δs_scaled, Δz_scaled, 0.99)
    @debug "Solution after step ok?: " is_convex_cone(program, program.α)
    @debug "Updated Step size: " * string(program.α)

    # update iterates with updated α
    status = solver.status
    push!(status.kkt_iterate, KKTIterate(program.KKT_x, program.s, program.z))
    update_iterates(program, Δs, Δx)

    # update scaling matrices and vars
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        Δs_scaled_k = @view Δs_scaled[inds]
        Δz_scaled_k = @view Δz_scaled[inds]
        update_scaling_vars(cone, Δs_scaled_k, Δz_scaled_k, program.α)
    end
    # end
    # println(update_elapsed_time)
    # @debug "Updated scaling matrices and scaling vars"
    # return result, r, μ
end

export ConeQP
export Solver
export TerminationStatus

export evaluate_optimality_conditions
export get_solution
export get_solver_status
export initialize!
export is_optimal
export run_solver
export update_equality_constraints
