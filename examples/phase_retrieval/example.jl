#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using FFTW
using LinearAlgebra
import Random

function get_problem_parameters(b, n)
    A = fft(I(n), 1)
    M = diagm(b) * (I - A*A') * diagm(b)
    idx = map(x -> CartesianIndex(x, x), 1:n)
    idx = lower_triangular_from_2d_idx(n, idx)
    idx = map(x -> CartesianIndex(x[1], x[2]), enumerate(idx))
    A = zeros((n, Int(n * (n + 1) / 2)))
    A[idx] .= 1
    b = ones(length(b))
    G = -Matrix(1.0I, size(A)[2], size(A)[2])
    P = zeros((size(A)[2], size(A)[2]))
    c = svec(M)
    h = zeros(size(G)[1])
    return A, G, P, b, c, h
end

function get_qp(b, n)
    A, G, P, b, c, h = get_problem_parameters(b, n)
    cones::Vector{Cone} = []
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{Float64, ComplexF64, ComplexF64}(A, G, P, b, c, h, cones)
    return cone_qp
end

function run_example()
    Random.seed!(1)
    n = 16
    b = rand(Float64, n)
    cone_qp = get_qp(b, n)
    solver = Solver(cone_qp)
    status = optimize!(solver)
    return status
end

run_example()