#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

module ConicSolveFR

using ConicSolve
using LinearAlgebra

mutable struct FacialReduction
    solver::ConicSolve.Solver
    FacialReduction() = new()
end

# Internal status codes, not to be exposed
@enum ReductionStatus begin
    ALREADY_FEASIBLE
    MINIMAL_FACE_IDENTIFICATION_FAILED
    NO_RESULT_BELOW_THRESHOLD
    REDUCED_PROBLEM_OPTIMAL
    REDUCED_PROBLEM_UNSOLVED
    REDUCTION_OTHER_ERROR
    WEAK_CONSTRAINT
end

# TODO: only handles PSD constraints!
function reduce_constraint(U::Matrix{Float64}, cone::NonNegativeOrthant, constraint::ConicExpression)
    println("NOT HANDLED")
    @assert false
end

function reduce_constraint(U::Matrix{Float64}, cone::SecondOrderCone, constraint::ConicExpression)
    println("NOT HANDLED")
    @assert false
end

function reduce_constraint(U::Matrix{Float64}, cone::PSDCone, constraint::ConicExpression, tol=1e-6)
    A_i = get_mat_from_lt_vec(constraint.lhs)
    reduced_A_i = U' * A_i * U
    reduced_a = get_vec_from_lt_mat(reduced_A_i)
    return reduced_a
end

function reduce_affine_constraints(in_program::ConeQP, out_program::ConeQP, U::Matrix{Float64}, cone::Cone, n)
    affine_constraints = find_affine_constraints_by_cone(in_program, cone)
    reduced_cone = add_variable(out_program, cone, n)
    for constraint in affine_constraints
        reduced_a = reduce_constraint(U, cone, constraint)
        add_affine_constraint(out_program, reduced_cone, reduced_a, constraint.rhs)
    end
    return reduced_cone
end

function reduce_inequality_constraints(in_program::ConeQP, out_program::ConeQP, U::Matrix{Float64}, cone::Cone)
    inequality_constraints = find_inequality_constraints_by_cone(in_program, cone)
    for constraint in inequality_constraints
        reduced_a = reduce_constraint(U, cone, constraint)
        add_inequality_constraint(out_program, cone, reduced_a, constraint.rhs)
    end
end

function reduce_objective(U::Matrix{Float64}, c::Vector{Float64})
    C = get_mat_from_lt_vec(c)
    reduced_C = U' * C * U
    reduced_c = get_vec_from_lt_mat(reduced_C)
    return reduced_c
end

function project_to_min_face(x::Vector{Float64}, in_program::ConeQP, out_program::ConeQP, cone::Cone, tol=1e-3)
    X = mat(x)
    F = svd(X)
    inds = findall(F.S .>= tol)
    if length(inds) == 0
        reduced_status = WEAK_CONSTRAINT
        @info "Constraint is weak, largest value $(maximum(F.S)) less than tolerance $(tol), dropping constraint"
        return nothing, nothing, false, reduced_status
    end
    U = F.U[:, inds]
    if length(inds) == length(F.S)
        reduced_status = ALREADY_FEASIBLE
        @info "Constraint already feasible, nothing to reduce"
        return nothing, nothing, false, reduced_status
    end

    reduced_cone = reduce_affine_constraints(in_program, out_program, U, cone, length(inds))
    return U, reduced_cone, true, REDUCED_PROBLEM_UNSOLVED
end

function expose_face(solver::ConicSolve.Solver)
    suppress_logging(solver)
    solver.program.c = zeros(length(solver.program.c))
    ConicSolve.run_solver(solver, false, false, false)
    program = solver.program
    A = program.A
    y_inds = program.inds_b
    y = program.KKT_x[y_inds]
    s = A' * y
    x_inds = program.inds_c
    s_perp = program.KKT_x[x_inds]
    return s, s_perp
end

function log_best_iterate(solver::ConicSolve.Solver, i)
    status = get_solver_status(solver)
    program = solver.program
    y, _ = get_constraint_dual(program)
    b_y = program.b' * y
    additional_data = ["b'y" b_y]
    log_iteration_status(status, i == 1, status.best_iterate, additional_data)
end

function reduce_cone(solver::ConicSolve.Solver, cone::Cone, orig_obj, tol=1e-1, truncation_tol=1e-3)
    # return facial reduced problem as per Algorithm 1.1 Permenter 2017
    Us = []
    U = nothing
    i = 1
    reduced_cone = typeof(cone)(cone.p)
    reduced_obj = orig_obj
    reduced_program = nothing
    reduced_status = REDUCED_PROBLEM_UNSOLVED
    while true
        # find s ∈ F* \ F\\^{\perp}
        s, s_perp = expose_face(solver)

        status = get_solver_status(solver)
        log_best_iterate(solver, i)
        if status.status_termination == ConicSolve.INFEASIBLE
            @info "Face reduced from n = $(cone.p) to $(reduced_cone.p)"
            break
        end

        if (-tol <= dot(s, s_perp) <= tol) == false
            @info "WARNING: gap s'z $(dot(s, s_perp)) above threshold $(tol)"
        end
        
        # set F = F ∩ s\\^{\perp}
        reduced_program = ConicSolve.ConeQP()
        U, reduced_cone, reduced, reduced_status = project_to_min_face(s_perp, solver.program, reduced_program, cone, truncation_tol)
        if !isnothing(U)
            push!(Us, U)
        end
        if reduced == false
            @info "Nothing reduced"
            break
        end
        
        add_default_inequality_constraint(reduced_program, reduced_cone)
        
        # last chance to compute reduced objective, don't use it in auxiliary problem, just return it
        reduced_obj = reduce_objective(U, reduced_obj)

        reduced_program = build_program(reduced_program)
        
        solver = Solver(reduced_program)
        solver.presolve_scaling_method = "ruiz"
        
        @info "Reduced problem, i = $(i)"
        if i == cone.p
            @info "Face reduced to limit"
            break
        end
        i += 1
    end

    return Us, i, reduced_cone, reduced_obj, reduced_program, reduced_status
end

function reduce_cone_program(orig_solver::ConicSolve.Solver, cone::Cone, store_iterates=false, tol=2e-2, truncation_tol=1e-3)
    reduced_status = nothing
    out_program = get_subproblem(orig_solver.program, cone)
    @info "Subproblem constructed"
    
    solver = ConicSolve.Solver(out_program)
    solver.presolve_scaling_method = "ruiz"
    Us, i, reduced_cone, reduced_obj, reduced_program, reduced_status = reduce_cone(solver, cone, out_program.c, truncation_tol)
    if reduced_status == ALREADY_FEASIBLE
        @info "Subproblem already feasible"
    end
    if reduced_status == WEAK_CONSTRAINT
        return nothing, reduced_status, nothing
    end
    if isnothing(Us) || i == 1
        @info "Subproblem face reduction unsuccessful"
        reduced_status = MINIMAL_FACE_IDENTIFICATION_FAILED
        return nothing, reduced_status, nothing
    end
    @info "Subproblem reduced"

    # set objective of reduced problem
    set_objective(reduced_program, reduced_cone, reduced_obj)
    reduced_program = build_program(reduced_program)

    # solve the reduced problem
    cone_subproblem_solver = ConicSolve.Solver(reduced_program)
    cone_subproblem_solver.tol_optimality = tol
    # suppress_logging(cone_subproblem_solver)
    ConicSolve.run_solver(cone_subproblem_solver, false, false, false)
    x = get_solution(cone_subproblem_solver)
    status = get_solver_status(cone_subproblem_solver)
    if status.status_termination != ConicSolve.INFEASIBLE
        reduced_status = REDUCED_PROBLEM_OPTIMAL
        log_best_iterate(cone_subproblem_solver, 1)
        @info "Solved reduced problem"
    end

    if store_iterates == true
        reproj_x = []
        for iterate in status.kkt_iterate
            @info "Reproject iterate"
            x_inds = cone_subproblem_solver.program.inds_c
            reproj_x_i = reproject_to_original_form(Us, iterate.KKT_x[x_inds])
            push!(reproj_x, reproj_x_i)
        end
        return reproj_x, reduced_status, cone_subproblem_solver
    end
    reproj_x = reproject_to_original_form(Us, x)
    return reproj_x, reduced_status, cone_subproblem_solver
end

function reproject_to_original_form(Us, x)
    # U1 * mat(U2 * mat(...) * U2') * U1'
    Us = reverse(Us)
    reproj_x = mat(x)
    for U in Us
        reproj_x = U * reproj_x * U'
    end
    reproj_x = svec(reproj_x)
    return reproj_x
end

"""
    run_fr_solver(solver, store_iterates, tol, truncation_tol)

Runs face reduction algorithm on the problem given by the solver.

# Parameters:
* `solver`: The solver used with the given problem
* `store_iterates`: Set to true to return a vector of solutions corresponding to each iterate
* `tol`: absolute tolerance to determine exposed face (using duality gap)
* `truncation_tol`: absolute tolerance to remove near redundant constraints

Returns the solutions for each subproblem and corresponding solvers used for each subproblem
"""
function run_fr_solver(solver::ConicSolve.Solver,
                       store_iterates=false,
                       tol=2e-2,
                       truncation_tol=1e-3)
    # perform facial reduction on solver problem
    problem = FacialReduction()
    problem.solver = solver
    in_program = solver.program
    apply_regularization(in_program)

    @info "### Executing face reduction ###"
    vars = in_program.vars
    x = []
    reduced_status = nothing
    reduced_solvers = []
    for i = 1:length(vars.cones)
        cone = vars.cones[i]
        @info "Performing face reduction on cone $(i) of $(length(vars.cones))"
        x_i, reduced_status, reduced_solver = reduce_cone_program(solver, cone, store_iterates, tol, truncation_tol)
        if !(reduced_status in [REDUCED_PROBLEM_OPTIMAL, WEAK_CONSTRAINT])
            @info "Face reduction unsuccessful $(reduced_status)"
            return
        end
        push!(x, x_i)
        push!(reduced_solvers, reduced_solver)
    end
    println("Face reduction completed")
    return x, reduced_solvers
end

export run_fr_solver

end
