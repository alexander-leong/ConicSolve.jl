#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""see CVXOPT Vandenberghe 2010 for Cone QP definition"""

include("./arrayutils.jl")
include("./debug.jl")
include("./kktsolver.jl")

using LinearAlgebra
using Logging

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

"""
    ConeQP

Represents a Conic Quadratic Program.
"""
mutable struct ConeQP{T<:Number, U<:Number, V<:Number}
    A::AbstractArray{Float64}
    G::AbstractArray{V}
    KKT_b::AbstractVector{V}
    KKT_x::AbstractVector{V}
    P::AbstractArray{Float64}
    b::AbstractVector{Float64}
    c::AbstractVector{U}
    h::AbstractVector{Float64}
    s::AbstractVector{V}
    z::AbstractVector{V}
    α::Float64
    
    cones::Vector{Cone}
    cones_inds::Vector{Int}
    cones_p::Vector{Int}

    inds_c::UnitRange{Int64}
    inds_b::UnitRange{Int64}
    inds_h::UnitRange{Int64}

    kktsystem::KKTSystem

    @doc"""
        ConeQP(A, G, P, b, c, h, cones)
    
    Constructs a new Conic Quadratic Program of the form:
    ```math
    \\begin{aligned}
    \\text{minimize}\\qquad &
    (1/2)x^TPx + c^Tx \\\\
    \\text{subject to}\\qquad &
    Gx + s = h \\\\
    & Ax = b \\\\
    & s \\succeq 0
    \\end{aligned}
    ```
    
    # Parameters:
    * `A`: The block matrix A in the KKT matrix
    * `G`: The block matrix G in the KKT matrix
    * `P`: The block matrix P in the KKT matrix
    * `b`: The vector b corresponding to ``Ax = b``
    * `c`: The vector c corresponding to ``c^Tx``
    * `h`: The vector h corresponding to ``Gx + s = h``
    * `cones`: A vector of cone types

    The cones vector is an ordered vector corresponding to the
    conic constraints defined by:
    ```math
    \\begin{aligned}
    Gx + s = h \\\\
    s \\succeq_K 0
    \\end{aligned}
    ```
    where ``\\succeq_K`` is a generalized inequality with respect to cone K.

    NOTE: The K is sometimes dropped to simplify notation.
    """
    function ConeQP{T, U, V}(A::AbstractArray{Float64},
                    G::AbstractArray{T},
                    P::AbstractArray{Float64},
                    b::AbstractArray{Float64},
                    c::AbstractArray{U},
                    h::AbstractArray{Float64},
                    cones::Vector{Cone}) where {T<:Number, U<:Number, V<:Number}
        cone_qp = new{T, U, V}()
        cone_qp.G = G
        cone_qp.P = P
        cone_qp.c = c
        cone_qp.h = h
        if A != undef
            if size(A)[2] != size(G)[2]
                throw(DimensionMismatch("Number of columns of A does not equal G"))
            else
                cone_qp.A = A
            end
        end
        if size(G)[2] != size(P)[2]
            throw(DimensionMismatch("Number of columns of G does not equal P"))
        end
        if size(P)[1] != size(P)[2]
            throw(DimensionMismatch("P is not square"))
        end
        cone_qp.inds_c = 1:size(P)[1]
        if b != undef
            if size(A)[1] != length(b)
                throw(DimensionMismatch("Number of rows of A does not equal b"))
            else
                cone_qp.b = b
                cone_qp.inds_b = cone_qp.inds_c[end]+1:cone_qp.inds_c[end]+size(A)[1]
                cone_qp.inds_h = cone_qp.inds_b[end]+1:cone_qp.inds_b[end]+size(G)[1]
            end
        else
            cone_qp.inds_h = cone_qp.inds_c[end]+1:cone_qp.inds_c[end]+size(G)[1]
        end
        if size(A)[2] != length(c)
            throw(DimensionMismatch("Number of columns of A does not equal c"))
        end
        if size(G)[1] != length(h)
            throw(DimensionMismatch("Number of rows of G does not equal h"))
        end
        cone_qp.cones = cones
        
        return cone_qp
    end
    
    @doc"""
        ConeQP(G, P, c, h, cones)
    
    Constructs a new Conic Quadratic Program of the form:
    ```math
    \\begin{aligned}
    \\text{minimize}\\qquad &
    (1/2)x^TPx + c^Tx \\\\
    \\text{subject to}\\qquad &
    Gx + s = h \\\\
    & s \\succeq 0
    \\end{aligned}
    ```
    
    # Parameters:
    * `G`: The block matrix G in the KKT matrix
    * `P`: The block matrix P in the KKT matrix
    * `c`: The vector c corresponding to ``c^Tx``
    * `h`: The vector h corresponding to ``Gx + s = h``
    * `cones`: A vector of cone types
    """
    function ConeQP(G::AbstractArray{Float64},
                    P::AbstractArray{Float64},
                    c::AbstractArray{Float64},
                    h::AbstractArray{Float64},
                    cones::Vector{Cone})
        cone_qp = ConeQP(undef, G, P, undef, c, h, cones)
        return cone_qp
    end
end

mutable struct SolverStatus
    current_iteration::Int32
    duality_gap::AbstractArray{Float64}
    residual_x::AbstractArray{Float64}
    residual_y::AbstractArray{Float64}
    residual_z::AbstractArray{Float64}
    status_termination::Union{TerminationStatus, Nothing}

    function SolverStatus()
        status = new()
        status.current_iteration = 0
        status.duality_gap = []
        status.residual_x = []
        status.residual_y = []
        status.residual_z = []
        status.status_termination = OPTIMIZE_NOT_CALLED
        return status
    end
end

"""
    Solver

Represents an Interior Point Method solver
for solving Conic Quadratic Programs.
"""
mutable struct Solver
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
    status_dual
    status_primal
    status_termination::Union{TerminationStatus, Nothing}
    time_limit_sec::Int32
    tol_gap_abs::Union{Float64, Nothing}
    tol_gap_rel::Union{Float64, Nothing}
    tol_optimality::Union{Float64, Nothing}
    η
    γ::Float64 # Mehrotra correction parameter

    @doc"""
        Solver(program, kktsolve, η, γ)
    
    Constructs a new QP solver object.
    
    # Parameters:
    * `program`: The QP to solve
    * `kktsolve`: The function to solve the KKT system
    * `limit_obj`: The minimum/maximum objective value the solver will terminate
    * `limit_soln`: The 2-norm difference between the current and previous estimates
    * `tol_gap_abs`: The absolute gap tolerance
    * `tol_gap_rel`: The relative gap tolerance
    * `tol_optimality`: The absolute tolerance for satisfying the optimality conditions
    * `max_iterations`: The maximum number of iterations before the solver terminates
    * `time_limit_sec`: The elapsed time before terminating the solver after the current iteration
    * `η`: Optimization parameter typically set to zero or σ, default is 0.0
    * `γ`: Mehrotra correction parameter set to γ ∈ [0, 1], default is 1.0

    NOTE: Set η as nothing to set to σ.
    """
    function Solver(program::ConeQP,
                    kktsolve="qrchol",
                    limit_obj=0,
                    limit_soln=0,
                    tol_gap_abs=1e-6,
                    tol_gap_rel=1e-6,
                    tol_optimality=1e-6,
                    max_iterations=100,
                    time_limit_sec=1e6,
                    η=nothing,
                    γ::Float64=1.0)
        solver = new()
        solver.current_iteration = 1
        solver.device = CPU
        solver.kktsolver = setup_default_kkt_solver(kktsolve)
        solver.program = program
        solver.limit_obj = limit_obj
        solver.limit_soln = limit_soln
        solver.max_iterations = max_iterations
        solver.num_threads = 1
        solver.status = SolverStatus()
        solver.time_limit_sec = time_limit_sec
        solver.tol_gap_abs = tol_gap_abs
        solver.tol_gap_rel = tol_gap_rel
        solver.tol_optimality = tol_optimality
        solver.η = η
        solver.γ = γ
        return solver
    end
end

function within_tol(abs_tol, rel_tol, value)
    if abs_tol / value < rel_tol
        return true
    end
    return false
end

"""
    get_solution(solver)

Get the solution to the optimization problem.
"""
# FIXME: this is broken
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

"""
    optimize!(solver)

Executes the solver on the optimization problem.
"""
function optimize!(solver::Solver, is_init=false)
    try
        print_graphic()
        @info "Optimize called"
        if is_init == false
            initialize!(solver)
            check_preconditions(solver)
        end
        result = optimize_main!(solver)
        log_msg = ("Solver finished" * "\n" *
            "Exit status: " * string(solver.status.status_termination) * "\n" *
            "Primal objective value: " * string(solver.obj_primal_value) * "\n" *
            "Number of iterations: " * string(solver.current_iteration - 1) * "\n" *
            "Time elapsed: " * string(solver.solve_time))
        @info log_msg
        return result
    catch err
        if isa(err, LAPACKException)
            solver.status.status_termination = NUMERICAL_ERROR
        end
    end
end

"""
    get_objective(P, c, x)

Get the current objective value to the conic quadratic program
`` x^TPx + c^Tx ``
"""
function get_objective(P::AbstractArray{Float64},
                       c::AbstractArray{U},
                       x::AbstractArray{T}) where {T<:Number, U<:Number}
    return x' * P * x + c' * x
end

function check_preconditions(solver::Solver)
    qp = solver.program
    if isdefined(qp, :A) && rank(qp.A) < size(qp.A)[1]
        solver.status.status_termination = INFEASIBLE
        @error "Values of A are inconsistent or redundant."
        @assert false
    end
    if isdefined(qp, :A)
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

function optimize_main!(solver::Solver)
    # using Conic QP (Primal-Dual Interior Point) method
    # given strictly feasible x, t := t^(0) > 0, mu > 1, tolerance eps > 0.
    solver.obj_primal_value = Inf
    total_time_elapsed = 0
    @info "Executing main optimization loop"
    while true
        i = solver.current_iteration
        itr_time_elapsed = @elapsed begin
        program = solver.program
        P = program.P
        c = program.c
        x = @view program.KKT_x[1:size(P)[1]]
        primal_obj = get_objective(P, c, x)
        current_obj = solver.obj_primal_value
        solver.obj_primal_value = primal_obj
        result, r, μ = get_solver_status(solver)
        status = solver.status
        if i > 1 && (P != zeros(size(P)) || c != zeros(size(P)[1]))
            if abs(solver.obj_primal_value) <= solver.limit_obj
                solver.status.status_termination = OBJECTIVE_LIMIT
                return status
            end
            if abs(current_obj - solver.obj_primal_value) < solver.tol_gap_abs
                solver.status.status_termination = SLOW_PROGRESS
                return status
            end
        end
        if i > solver.max_iterations
            solver.status.status_termination = ITERATION_LIMIT
            return status
        end
        if result == true
            solver.status.status_termination = OPTIMAL
            return status
        end
        η = solver.η
        γ = solver.γ
        get_central_path(solver, i, r, μ, η, γ)
        end
        ### TEST CODE ###
        if i == 10
            return
        end
        ### TEST CODE ###
        solver.current_iteration += 1
        total_time_elapsed = total_time_elapsed + itr_time_elapsed
        solver.solve_time = total_time_elapsed
        if total_time_elapsed > solver.time_limit_sec
            solver.status.status_termination = TIME_LIMIT
            break
        end
    end
end

function print_table(data, i, n=100, pad=18)
    if i % n == 1
        row = "| "
        for val in eachrow(data)
            row *= lpad(val[:][1], pad, " ") * " |"
        end
        println(row)
    end
    row = "| "
    for val in eachrow(data)
        row *= lpad(round(val[:][end], digits=9), pad, " ") * " |"
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
                              result::Bool) where T <: Number
    data = ["Iter." i;
    "Dual Res. (x)" norm(r_x);
    "Primal Res. (y)" norm(r_y);
    "Cent. Res. (z)" norm(r_z);
    "Gap" norm(g);
    "Primal Obj." pri_obj;]
    # TODO
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
    num_constraints = size(program.P)[1]
    if isdefined(program, :A)
        num_constraints += size(program.A)[1]
    end
    num_constraints += size(program.G)[1]
    return num_constraints
end

function initialize!(solver::Solver)
    solver.status.status_termination = OPTIMIZE_NOT_CALLED
    log_solver_parameters(solver)
    program = solver.program
    A = isdefined(program, :A) ? program.A : nothing
    G = program.G
    P = program.P

    program.kktsystem = isdefined(program, :A) ? KKTSystem(A, G, P) : KKTSystem(G, P)
    if isdefined(program, :b)
        program.KKT_b = vcat([-program.c, program.b, program.h]...)
    else
        program.KKT_b = vcat([-program.c, program.h]...)
    end
    program.s = zeros((size(program.G)[1]))
    program.z = zeros((size(program.G)[1]))

    # initialize cones
    e = []
    program.cones_inds = [0]
    for (_, cone) in enumerate(program.cones)
        e = vcat(e, get_e(cone))
        ind = program.cones_inds[end] + get_size(cone)
        push!(program.cones_inds, ind)
    end

    @info "Solving for primal dual starting points"
    inv_W = Matrix{Float64}(I, size(G)[1], size(G)[1])
    G_scaled = -inv_W' * program.G
    b_z = @view program.KKT_b[program.inds_h]
    inv_W_b_z = -inv_W * b_z
    kkt_1_1 = program.P + (G_scaled' * G_scaled)
    x_hat = qp_solve(solver, G_scaled, kkt_1_1, inv_W_b_z, full_qr_solve)
    x_inds = 1:size(P)[1]
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

function check_linear_equalities(program::ConeQP,
                                 r::Vector{T},
                                 tol::Float64=1e-6) where T <: Number
    y_inds = program.inds_b
    z_inds = program.inds_h
    b_y = -r[y_inds]
    b_z = -r[z_inds] - program.s
    b_y_is_optimal = all(x->abs(x) <= tol, b_y)
    b_z_is_optimal = all(x->abs(x) <= tol, b_z)
    return b_y_is_optimal && b_z_is_optimal
end

function check_linear_inequalities(s::Vector{T},
                                   z::AbstractArray,
                                   tol::Float64=1e-6) where T <: Number
    r2 = all(s->norm(s) >= tol, s) && all(z->norm(z) >= tol, z)
    return r2
end

function check_duality_gap(s::Vector{T},
                           z::AbstractArray,
                           atol::Float64=1e-6,
                           rtol::Float64=1e-3) where T <: Number
    r3 = within_tol(atol, rtol, abs(z' * s))
    return r3
end

"""
    is_optimal(program, r, s, z, gap_atol, gap_rtol, tol, check_r1, check_r2, check_r3, check_r4)

The stopping criterion used to determine convergence.
Convergence is based on the following criteria:
- residuals close to zero (within tol)
- duality gap close to zero (within tol)
- solution is primal-dual feasible

Certain optimality conditions may be toggled by the check_* parameters,
though this is discouraged.
"""
function is_optimal(program::ConeQP,
                    r::AbstractArray{T},
                    s::AbstractArray{T},
                    z::AbstractArray{T},
                    gap_atol::Float64,
                    gap_rtol::Float64,
                    tol::Float64,
                    check_r1=true,
                    check_r2=true,
                    check_r3=true,
                    check_r4=true) where T <: Number
    x_inds = program.inds_c
    b_x = r[x_inds]
    b_x_is_optimal = all(x->abs(x) <= tol, b_x)
    x = program.KKT_x
    
    r1 = check_r1 ? b_x_is_optimal : true
    check_r2 = check_r2 && isdefined(program, :A)
    r2 = check_r2 ? check_linear_equalities(program, r) : true
    r3 = check_r3 ? check_linear_inequalities(s, z) : true
    r4 = check_r4 ? check_duality_gap(s, z, gap_atol, gap_rtol) : true
    r5 = is_convex_cone(program, 0, tol)

    result = r1 && r2 && r3 && r4 && r5
    return result
end

function get_inv_weighted_mat(program::ConeQP,
                              V::AbstractArray{T},
                              transpose=false) where T <: Number
    ncols = length(size(V)) == 1 ? 1 : size(V)[2]
    inv_W_V = zeros(T, (size(V)[1], ncols))
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        inv_W_V[inds, :] = get_inv_weighted_mat(cone, V[inds, :], transpose)
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
        W_V[inds, :] = get_weighted_mat(cone, V[inds, :])
        if update_var == true
            cone.λ = W_V[inds, 1]
        end
    end
    return W_V
end

function evaluate_residual(qp::ConeQP,
                           Δx::Vector{T},
                           x_inds::UnitRange{Int64},
                           y_inds::UnitRange{Int64},
                           z_inds::UnitRange{Int64}) where T <: Number
    r = zeros(T, length(Δx))
    x = Δx[x_inds]
    z = Δx[z_inds]
    if isdefined(qp, :A)
        y = Δx[y_inds]
        r_x = qp.P * x + qp.A' * y + qp.G' * z + qp.c
        r_y = qp.A * x - qp.b
        r[y_inds] = r_y
    else
        r_x = qp.P * x + qp.G' * z + qp.c
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
    s = program.s
    Δx = @view x[x_inds]
    b_z = program.G * Δx - b
    b_z_scaled = get_inv_weighted_mat(program, b_z)
    Δz = get_inv_weighted_mat(program, b_z_scaled, true)
    Δs = -s - b_z
    x[z_inds] = Δz[:, 1]
    Δz_scaled = b_z_scaled
    Δs_scaled = get_inv_weighted_mat(program, Δs)
    @debug "Calculated affine direction"
    return Δs, Δs_scaled, Δz_scaled, x
end

function get_combined_direction(program::ConeQP,
                                d_z::AbstractArray,
                                x::AbstractArray)
    G = program.G
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
    Δx = qp_solve(solver, G_scaled, kkt_1_1, inv_W_b_z, kktsolver_fn)
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
    Δx = qp_solve(solver, G_scaled, kkt_1_1, inv_W_b_z, kktsolver_fn)
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

function get_solver_status(solver::Solver)
    i = solver.current_iteration
    gap_atol = solver.tol_gap_abs
    gap_rtol = solver.tol_gap_rel
    tol = solver.tol_optimality
    program = solver.program
    x = program.KKT_x
    s = program.s

    # range objects for indexing
    x_inds = program.inds_c
    y_inds = program.inds_b
    z_inds = program.inds_h
    
    # evaluate residual
    r = evaluate_residual(program, x, x_inds, y_inds, z_inds)
    r_x = @view r[x_inds]
    if isdefined(program, :A)
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

    # evaluate stopping criteria
    z = @view x[z_inds]
    result = is_optimal(program, r, s, z, gap_atol, gap_rtol, tol)
    pri_obj = solver.obj_primal_value
    log_iteration_status(i, r_x, r_y, r_z, μ, pri_obj, result)

    # save status
    status = solver.status
    status.current_iteration = i
    push!(status.duality_gap, norm(μ))
    push!(status.residual_x, norm(r_x))
    push!(status.residual_y, norm(r_y))
    push!(status.residual_z, norm(r_z))

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
    λ = zeros(length(program.s))
    if current_itr == 1
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
    # solve linear equations to get affine direction
    # see page 14 of coneprog.pdf for setting z part of b
    s = program.s
    program.KKT_b = -r
    program.KKT_b[z_inds] = -r[z_inds] + s
    G_scaled = get_inv_weighted_mat(program, program.G)
    b_z = @view program.KKT_b[program.inds_h]
    inv_W_b_z = get_inv_weighted_mat(program, b_z)
    kkt_1_1 = program.P + (G_scaled' * G_scaled)
    Δsₐ, Δsₐ_scaled, Δzₐ_scaled, Δxₐ = kktsolver.affine_search_direction(solver, G_scaled, kkt_1_1, inv_W_b_z)
    update_cones(program, Δsₐ, Δxₐ[z_inds])

    @debug "Affine direction ok?: " check_affine_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled)

    # Compute step size and centering parameter
    α = get_step_size(program, Δsₐ_scaled, Δzₐ_scaled)
    @debug "Solution after step ok?: " is_convex_cone(program, α)
    ρ = 1 - α + α^2 * dot(Δsₐ_scaled', Δzₐ_scaled) / dot(λ', λ)
    σ = maximum((0, minimum((1, abs(ρ)))))^3
    if η === nothing
        η = σ
    end
    @debug "Step size: " α
    @debug "η: " η
    @debug "Centering parameter: " σ
    
    # Combined direction, i.e. solve linear equations
    # see page 29 of coneprog.pdf for solving a KKT system with SOCP and SDP constraints
    KKT_b = -(1 - η) * r
    b_z = copy(KKT_b[z_inds])
    # see eq. 19a, 19b, 22a of coneprog.pdf
    KKT_b_z = program.KKT_b[z_inds]
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        b_z_k = b_z[inds]
        Δsₐ_scaled_inds = Δsₐ_scaled[inds]
        Δzₐ_scaled_inds = Δzₐ_scaled[inds]
        d_s = get_d_s(cone, Δsₐ_scaled_inds, Δzₐ_scaled_inds, b_z_k, γ, λ[inds], μ, σ)
        KKT_b_z[inds] = d_s
    end
    Δs, Δs_scaled, Δz_scaled, Δx = kktsolver.combined_search_direction(solver, G_scaled, b_z, kkt_1_1, inv_W_b_z)
    update_cones(program, Δs, Δx[z_inds])

    @debug "Combined direction ok?:" check_combined_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled, Δs_scaled, Δz_scaled, μ, σ)
    
    # update step size
    program.α = get_step_size(program, Δs_scaled, Δz_scaled, 0.99)
    @debug "Solution after step ok?: " is_convex_cone(program, program.α)
    @debug "Updated Step size: " * string(program.α)

    # update iterates with updated α
    update_iterates(program, Δs, Δx)

    # update scaling matrices and vars
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        update_scaling_vars(cone, Δs_scaled[inds], Δz_scaled[inds], program.α)
    end
    @debug "Updated scaling matrices and scaling vars"
end

export ConeQP
export Solver
export optimize!