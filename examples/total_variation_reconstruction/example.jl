#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using LinearAlgebra
import Random

function run_example()
    Random.seed!(1)
    sig = sin.(0:0.1:8)[1:end-1]
    n = length(sig)
    b = sig .+ rand(0:0.1:1, n) # signal + random noise
    λ = 1 # smoothing parameter
    
    # construct 1-d finite difference matrix
    D = Matrix{Float64}(-1.0I, n-1, n)
    inds = map(x -> CartesianIndex(x, x+1), 1:n-1)
    D[inds] .= 1

    program = ConeQP()

    x = add_variable(program, NonNegativeOrthant(n), n)
    x_hat = add_variable(program, NonNegativeOrthant(n), n)

    define_program(program,
                minimize(l2(D, x_hat) + λ * l1(D, x_hat)),
                x == b,
                x_hat[1] == b[1])
    
    solver = Solver(program)
    solver.max_iterations = 20
    # status = run_solver(solver)
    # return x, cone_qp
end

run_example()