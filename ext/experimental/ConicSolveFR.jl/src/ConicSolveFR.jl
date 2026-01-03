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

function reduce_constraint(U::Matrix{Float64}, cone::PSDCone, constraint::ConeConstraint)
    a = get_constraint(program, cone, constraint.lhs)
    A_i = get_mat_from_lt_vec(a)
    reduced_A_i = U' * A_i * U
    reduced_a = get_vec_from_lt_mat(reduced_A_i)
    return reduced_a
end

function reduce_affine_constraints(in_program::ConeQP, out_program::ConeQP, U::Matrix{Float64}, cone::Cone)
    affine_constraints = find_affine_constraints_by_cone(in_program, cone)
    for constraint in affine_constraints
        reduced_a = reduce_constraint(U, cone, constraint)
        add_affine_constraint(out_program, cone, reduced_a, constraint.rhs)
    end
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

function project_to_min_face(U::Matrix{Float64}, in_program::ConeQP, out_program::ConeQP, cone::Cone)
    c = in_program.c
    reduced_c = zeros(length(c))

    U_k = get_elements_by_constraint_inds(in_program, cone, U)
    reduce_affine_constraints(in_program, out_program, U_k, cone)
    reduce_inequality_constraints(in_program, out_program, U_k, cone)
    inds = get_indices_of_constraint(in_program, cone)
    reduced_c = reduce_objective(U_k, c[inds])
    
    set_objective(out_program, reduced_c)
end

function expose_face(solver::ConicSolve.Solver, cone::Cone)
    ConicSolve.run_solver(solver)
    return nothing, nothing
    program = solver.program
    A = program.A
    y_inds = program.inds_b
    y = program.KKT_x[y_inds]
    s = cone.s
    s_perp = A' * y
    return s, s_perp
end

function update_face(solver::ConicSolve.Solver, cone::Cone, s_perp::Vector{Float64})
    constraint = ConeConstraint(cone, s_perp, 0)
    program = solver.program
    n = get_size(cone)
    v = zeros(n)
    v = get_constraint(program, constraint, v)
    G = program.G
    G = vcat(G, v')
    program.G = G
    push!(program.h, constraint.rhs)
    update_inequality_constraints(program)
    return constraint
end

function reduce_cone(solver::ConicSolve.Solver, cone::Cone, tol=1e-6)
    program = solver.program
    constraints::Vector{ConeConstraint} = []
    
    # return facial reduced problem as per Algorithm 1.1 Permenter 2017
    while true
        # step 1
        println("Expose face")
        s, z = expose_face(solver, cone)
        return nothing

        status = get_solver_status(solver)
        if status.status_termination == INFEASIBLE
            break
        end

        @assert isapprox(dot(s, z), tol)
        
        # step 2
        println("Update face")
        constraint = update_face(solver, cone, z)
        push!(constraints, constraint)
    end

    println("Get constraint matrix")
    U = get_constraint_matrix(program, constraints)
    return U
end

# function get_inequality_constraint(program::ConicSolve.ConeQP, constraint::ConeConstraint, cone::NonNegativeOrthant)
#     add_inequality_constraint(program, cone, constraint.lhs, constraint.rhs)
# end

# function get_inequality_constraint(program::ConicSolve.ConeQP, constraint::ConeConstraint, cone::SecondOrderCone)
#     add_inequality_constraint(program, cone, constraint.lhs, constraint.rhs)
# end

function get_inequality_constraint(program::ConicSolve.ConeQP, constraint::ConeConstraint, cone::PSDCone)
    add_inequality_constraint(program, cone, constraint.lhs, constraint.rhs)
    num_vars = get_size(cone)

    # add nonnegative constraint on matrix diagonal to improve solver robustness
    nonneg_cone = add_variable(program, NonNegativeOrthant(), cone.p)
    lhs = -Matrix{Float64}(I, cone.p, cone.p)
    lower_bound_nonnegativity_slack = 0.0
    constraint = add_affine_constraint(program, nonneg_cone, lhs, lower_bound_nonnegativity_slack * ones(cone.p))
    lhs = zeros((cone.p, num_vars))
    inds = get_diagonal_idx(cone.p)
    for i in 1:cone.p
        lhs[i, inds[i]] = 1
    end
    set_affine_constraint(constraint, cone, lhs)
end

function get_subproblem(in_program::ConicSolve.ConeQP, cone::Cone)
    # given initial problem, in_program, construct subproblem from constraints and objective with respect to given cone
    out_program = ConicSolve.ConeQP()

    affine_constraints = find_affine_constraints_by_cone(in_program, cone)
    for constraint in affine_constraints
        add_affine_constraint(out_program, cone, constraint.lhs, constraint.rhs)
    end

    # TODO how to add existing inequality constraints without redundancy
    # inequality_constraints = find_inequality_constraints_by_cone(in_program, cone)
    # for constraint in inequality_constraints
    #     get_inequality_constraint(out_program, constraint, cone)
    # end

    inds = get_indices_of_constraint(in_program, cone)
    # obj = in_program.c[inds]
    obj = zeros(length(inds))
    set_objective(out_program, cone, obj)

    add_variable(out_program, cone, cone.p)

    out_program = build_program(out_program, true)
    return out_program
end

function reduce_cone_program(orig_solver::ConicSolve.Solver, cone::Cone)
    out_program = get_subproblem(orig_solver.program, cone)
    solver = ConicSolve.Solver(out_program)
    println("Subproblem constructed")
    U = reduce_cone(solver, cone)
    return
    minface_program = ConicSolve.ConeQP()
    project_to_min_face(U, out_program, minface_program, cone)
    cone_subproblem_solver = ConicSolve.Solver(minface_program)
    ConicSolve.run_solver(cone_subproblem_solver)
end

function run_fr_solver(solver::ConicSolve.Solver)
    # perform facial reduction on solver problem
    problem = FacialReduction()
    problem.solver = solver
    in_program = solver.program

    for (i, cone) in enumerate(in_program.cones)
        println("Performing face reduction on cone $(i)")
        reduce_cone_program(solver, cone)
        return
    end
    println("Done!")
end

export run_fr_solver

end
