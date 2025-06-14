#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using LinearAlgebra

"""
    discretize()

Performs trapezoidal collocation on the given variable.
"""
function discretize()
end

"""
    get_state_space_matrix()

Get the SDP relaxed submatrix of the linearized vehicle dynamics eqns.
"""
function get_state_space_matrix()
end

"""
    set_initial_final_conditions()

Returns the ``A`` matrix and ``b`` vector that sets the initial and final conditions.
"""
function set_initial_final_conditions()
end

function get_problem_parameters()
end

function get_qp()
end

function run_example()
    D = 0
    I_sp = 0
    L = 0
    T = 0

    # initial conditions
    h_0 = 0
    m_0 = 0
    r_0 = 0
    v_0 = 0
    γ_0 = 0

    # final conditions
    h_f = 0
    v_f = 0
    γ_f = 0
end

run_example()