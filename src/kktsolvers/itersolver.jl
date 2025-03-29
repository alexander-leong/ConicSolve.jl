#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra
using Logging

function get_kkt_matrix(kktsystem, b_x, b_y, b_z)
    A = kktsystem.A
    G = kktsystem.G
    P = kktsystem.P
    b = vcat(b_x, b_y, b_z)
    x_0 = zeros(length(b))
    S_0 = [P A' G']
    dim = (size(A)[1], length(b_y) + length(b_z))
    S_1 = hcat(A, zeros(dim))
    dim = (size(G)[1], length(b_y))
    S_2 = hcat(G, zeros(dim), Matrix{Float64}(I, length(b_z), length(b_z)))
    S = [S_0; S_1; S_2]
    return S, b, x_0
end

export get_kkt_matrix