#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("polyutils.jl")

using ConicSolve
using LinearAlgebra
using Polynomials

function get_traj_qp()
    # use fourth order polynomial
    s = Polynomial([1, 1, 1, 1, 1], :t)
    sos = SOS(length(s))
    s_sos = s
    v_sos = derivative(s_sos)

    # set initial x-position at t = 0
    p = evaluate_monomials(s_sos, 0)
    add_polynomial_equality_constraint(sos, 0, p)
    # set final x-position at t = 1
    p = evaluate_monomials(s_sos, 1)
    add_polynomial_equality_constraint(sos, 5, p)
    # set initial x-velocity at t = 0 less than or equal to 5
    p = evaluate_monomials(v_sos, 0)
    add_polynomial_inequality_constraint(sos, 5.5, p)
    # set x-velocity less than or equal to 5
    p = evaluate_monomials(v_sos, 1)
    add_polynomial_inequality_constraint(sos, 1.5, p)
    
    num_vars = size(sos.A)[2]
    # set objective as min. jerk
    P = zeros((num_vars, num_vars))
    inds = get_inds(sos, 4, 2)
    P[inds] .= 1
    c = zeros(num_vars)
    set_objective(sos, P, c)

    cone_qp = sos_to_qp(sos)
    return cone_qp
end

function run_example()
    cone_qp = get_traj_qp()

    # solve optimization problem
    solver = Solver(cone_qp)
    solver.max_iterations = 20
    status = optimize!(solver)
    return status
    # x = get_solution(solver)
end

# run_example()