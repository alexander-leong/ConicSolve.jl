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

# TODO: only handles PSD constraints!
function reduce_constraint(U::Matrix{Float64}, cone::NonNegativeOrthant, constraint::ConeConstraint)
    println("NOT HANDLED")
    @assert false
end

function reduce_constraint(U::Matrix{Float64}, cone::SecondOrderCone, constraint::ConeConstraint)
    println("NOT HANDLED")
    @assert false
end

function reduce_constraint(U::Matrix{Float64}, cone::PSDCone, constraint::ConeConstraint, tol=1e-6)
    A_i = get_mat_from_lt_vec(constraint.lhs)
    reduced_A_i = U' * A_i * U
    reduced_a = get_vec_from_lt_mat(reduced_A_i)
    return reduced_a
end

function reduce_affine_constraints(in_program::ConeQP, out_program::ConeQP, U::Matrix{Float64}, cone::Cone, n)
    affine_constraints = find_affine_constraints_by_cone(in_program, cone)
    reduced_cone = add_variable(out_program, PSDCone(), n)
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
    println("Condition number $(cond(X))")
    println("Numerical rank $(length(inds))")
    U = F.U[:, inds]

    reduced_cone = reduce_affine_constraints(in_program, out_program, U, cone, length(inds))
    inds = get_indices_of_constraint(in_program, cone)
    return U, reduced_cone
end

function expose_face(solver::ConicSolve.Solver)
    suppress_logging(solver)
    ConicSolve.run_solver(solver)
    program = solver.program
    A = program.A
    y_inds = program.inds_b
    y = program.KKT_x[y_inds]
    s = A' * y
    x_inds = program.inds_c
    s_perp = program.KKT_x[x_inds]
    return s, s_perp
end

function reduce_cone(solver::ConicSolve.Solver, cone::Cone, tol=1e-1)
    # return facial reduced problem as per Algorithm 1.1 Permenter 2017
    U = nothing
    i = 1
    reduced_cone = typeof(cone)(cone.p)
    reduced_program = nothing
    while true
        # find s ∈ F* \ F\\^{\perp}
        s, s_perp = expose_face(solver)

        status = get_solver_status(solver)
        log_iteration_status(solver, i == 1, status.best_iterate)
        if status.status_termination == ConicSolve.INFEASIBLE || i == 2
            break
        end

        if (-tol <= dot(s, s_perp) <= tol) == false
            @info "WARNING: gap s'z $(dot(s, s_perp)) above threshold $(tol)"
        end
        
        # set F = F ∩ s\\^{\perp}
        reduced_program = ConicSolve.ConeQP()
        U, reduced_cone = project_to_min_face(s_perp, solver.program, reduced_program, cone)
        reduced_program = build_program(reduced_program)
        solver = Solver(reduced_program)
        @info "Reduced problem, i = $(i)"
        i += 1
    end

    return U, reduced_cone, reduced_program
end

function get_subproblem(in_program::ConicSolve.ConeQP, cone::Cone)
    # given initial problem, in_program, construct subproblem from constraints and objective with respect to given cone
    out_program = ConicSolve.ConeQP()

    affine_constraints = find_affine_constraints_by_cone(in_program, cone)
    for constraint in affine_constraints
        add_affine_constraint(out_program, cone, constraint.lhs, constraint.rhs)
    end

    inds = get_indices_of_constraint(in_program, cone)
    # obj = in_program.c[inds]
    # inds = get_size(cone)
    obj = zeros(length(inds))
    set_objective(out_program, cone, obj)

    add_variable(out_program, cone, cone.p)

    out_program = build_program(out_program, true)
    return out_program
end

function reduce_cone_program(orig_solver::ConicSolve.Solver, cone::Cone)
    out_program = get_subproblem(orig_solver.program, cone)
    @info "Subproblem constructed"
    
    solver = ConicSolve.Solver(out_program)
    solver.presolve_scaling_method = "ruiz"
    U, reduced_cone, reduced_program = reduce_cone(solver, cone)
    if isnothing(U)
        @info "Subproblem face reduction unsuccessful"
        return nothing
    end
    @info "Subproblem reduced"

    # set objective of reduced problem
    c = out_program.c
    reduced_c = reduce_objective(U, c)
    set_objective(reduced_program, reduced_cone, reduced_c)
    return

    # solve the reduced problem
    cone_subproblem_solver = ConicSolve.Solver(reduced_program)
    ConicSolve.run_solver(cone_subproblem_solver)
    x = get_solution(cone_subproblem_solver)

    # reproject back to original form
    reproj_x = U * mat(x) * U'
    reproj_x = svec(reproj_x)
    return reproj_x
end

function run_fr_solver(solver::ConicSolve.Solver)
    # perform facial reduction on solver problem
    problem = FacialReduction()
    problem.solver = solver
    in_program = solver.program
    apply_regularization(in_program)

    @info "### STEP 3: FACE REDUCTION ###"
    x = zeros(in_program.cones_inds[end])
    # TODO: make multithreaded
    for i = 1:length(in_program.cones)
        cone = in_program.cones[i]
        @info "Performing face reduction on cone $(i)"
        x_i = reduce_cone_program(solver, cone)
        if isnothing(x_i)
            @info "Face reduction unsuccessful"
            return
        end
        return
        inds = in_program.cones_inds[i]:in_program.cones_inds[i+1]
        x[inds] = x_i
    end
    println("Done!")
end

export run_fr_solver

end
