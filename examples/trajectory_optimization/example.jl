#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

# include("polyutils.jl")

using ConicSolve
using LinearAlgebra
using DynamicPolynomials

function get_traj_qp()
    # use fourth order polynomial
    p = DynamicPolynomials.Polynomial([1, 1, 1, 1, 1], :t)
    sos = SOS(length(p))
    dp = ∂(p)
    ddp = ∂(dp)
    dddp = ∂(ddp)

    # set initial x-position at t = 0
    add_polynomial_equality_constraint(sos, 0, p, 0)
    # set final x-position at t = 1
    add_polynomial_equality_constraint(sos, 5, p, 1)
    # set initial x-velocity at t = 0 less than or equal to 5.5
    add_polynomial_inequality_constraint(sos, 5.5, dp, 0)
    # set final x-velocity at t = 1 less than or equal to 1.5
    add_polynomial_inequality_constraint(sos, 1.5, dp, 1)
    
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
    @polyvar t
    @polyvar q[1:4]

    p = q[4]*t^4 + q[3]*t^3 + q[2]*t^2 + q[1]*t
    # p = t^4 + t^3 + t^2 + t
    dp = differentiate(p, t)
    ddp = differentiate(dp, t)
    dddp = differentiate(ddp, t)

    program = ConeQP()
    sos = add_variable(program, SOS, p, t)

    # set initial x-position at t = 0 to 0
    # set final x-position at t = 1 to 5
    # set initial x-velocity at t = 0 less than or equal to 5.5
    # set final x-velocity at t = 1 less than or equal to 1.5
    p_initial = polynomial_substitute(p, t=>0)
    p_final = polynomial_substitute(p, t=>1)
    dp_initial = polynomial_substitute(dp, t=>0)
    dp_final = polynomial_substitute(dp, t=>1)

    jerk = dddp
    
    define_program(program,
                minimize(jerk),
                SOSPolynomial(p_initial, sos) == 0.,
                SOSPolynomial(p_final, sos) == 5.,
                SOSPolynomial(dp_initial, sos) <= 5.5,
                SOSPolynomial(dp_final, sos) <= 1.5)
    
    program = build_program(program)
    solver = Solver(program)
    # solver.max_iterations = 20
    # status = run_solver(solver)
    # return status
    # x = get_solution(solver)
end

run_example()