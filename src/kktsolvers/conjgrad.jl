#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("itersolver.jl")

using LinearAlgebra
using Logging

function conj_grad(A, b, x_0, eps=1e-3, max_iterations=Inf)
    i = 1
    x = x_0
    r = b - A*x
    d = r
    n = minimum((size(A)[1], max_iterations))
    @debug "Executing Conjugate Gradient method"
    r_abs = norm(r, 2)
    while i <= n && r_abs > eps
        @debug "||r||_2: " r_abs
        δ = r' * r
        α = δ / (d' * A * d)
        x = x + α * d
        r = r - α * A * d
        β = r' * r / δ
        d = r + β * d
        r_abs = norm(r, 2)
        i = i + 1
    end
    return x
end

function conj_grad_kkt_solve(kktsystem, b_x, b_y, b_z)
    S, b, x_0 = get_kkt_matrix(kktsystem, b_x, b_y, b_z)
    x = conj_grad(S, b, x_0)
    return x
end

export conj_grad_kkt_solve