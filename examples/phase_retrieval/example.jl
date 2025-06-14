#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

""" WARNING: this example is computationally intensive, to get decent
results it is strongly recommended this problem be run on a machine with sufficient compute and memory capacity.
i.e. Memory requirements: >= 64GB RAM/VRAM
"""

using ConicSolve
# Uncomment to plot result
# using CairoMakie
using Distributions
using FFTW
using JLD
using LinearAlgebra
using ToeplitzMatrices
import Random

function complex_to_real_symmetric(A)
    A_Re = real(A)
    A_Im = imag(A)
    A = [A_Re -A_Im; A_Im A_Re]
    return A
end

function get_problem_parameters(A, pinv_A, b, n)
    M = Hermitian(diagm(b) * (I - (A * pinv_A)) * diagm(b))
    M = complex_to_real_symmetric(M)
    idx = map(x -> CartesianIndex(x, x), 1:n)
    idx = lower_triangular_from_2d_idx(n, idx)
    idx = map(x -> CartesianIndex(x[1], x[2]), enumerate(idx))
    A = zeros((n, Int(n * (n + 1) / 2)))
    A[idx] .= 1
    b = ones(n)
    G = -1.0I(size(A)[2])
    P = zeros((size(A)[2], size(A)[2]))
    c = svec(M)
    println(size(c))
    h = zeros(size(G)[1])
    return A, G, nothing, b, c, h
end

function warm_start_qp(data, n)
    idx = map(x -> CartesianIndex(x, x), 1:n)
    idx = lower_triangular_from_2d_idx(n, idx)
    idx = map(x -> CartesianIndex(x[1], x[2]), enumerate(idx))

    A = zeros((n, Int(n * (n + 1) / 2)))
    A[idx] .= 1
    b = ones(n)
    G = -1.0I(size(A)[2])
    P = nothing
    c = data["c"]
    h = zeros(size(G)[1])

    cones::Vector{Cone} = []
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)

    s = data["s"]
    z = data["z"]

    cone_qp.KKT_x = data["KKT_x"]
    cone_qp.s = s
    cone_qp.z = z
    return cone_qp
end

function get_qp(A, pinv_A, b, n)
    A, G, P, b, c, h = get_problem_parameters(A, pinv_A, b, n)
    cones::Vector{Cone} = []
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end

function get_filter_matrix(m, n)
    F = fft(I(Int(n)), 1)
    d = Normal(0, 0.01) # Normal random variable with mean, variance
    A = vcat([Matrix{ComplexF32}(Circulant(F * rand(d, n))) for i = 1:m]...)
    return A
end

function reconstruct_signal(u, b, pinv_A)
    # get the approximate phase vector from the solution
    U = mat(u)
    X = U[1:m*n, 1:m*n] + (U[m*n+1:end, 1:m*n] * im)
    s, v = eigen(X)
    u = v[:, end]

    # check SDP tightness to ensure quality of reconstruction
    println(s) # s should have only one large eigenvalue for X to be close to rank 1 to be tight

    # reconstruct the signal from the phase vector
    x = pinv_A * diagm(b) * u
    println(b - abs.(A * x))
    return x
end

function plot_result(input_signal, rec_signal, n)
    f = Figure()
    x_range = 1:1:n
    ax = Axis(f[1, 1], title="Input Signal vs Reconstructed Signal over time")
    lines!(ax, x_range, input_signal[1:n])
    lines!(ax, x_range, rec_signal[1:n])
    display(f)
end

Random.seed!(1)

# set parameters
m = 2 # number of illumination filters, >= 4 should be reasonable
n = 9 # must be large, say at least 60 which is intractable for most machines
x = collect(0:0.8:6.4) # smaller step size and longer length is better

# get the 1-D DFT matrix with n roots of unity
A = get_filter_matrix(m, n)

# create noiseless raw input signal
ω = 1
ϕ = 0
x_sig = sin.(ω .* x .+ ϕ)[1:end]

# calculate measurements by passing original signal through each filter
b = abs.(A * x_sig)

pinv_A = inv(A'*A) * A' # Moore Penrose Psedoinverse

cone_qp = get_qp(A, pinv_A, b, length(b)*2)
solver = Solver(cone_qp)
solver.device = GPU
solver.max_iterations = 10
solver.num_threads = 1
status = run_solver(solver)

# evaluate reconstruction accuracy
u = get_solution(solver)
rec_signal = reconstruct_signal(u, b, pinv_A)
# plot_result(x_sig, rec_signal, length(x_sig))

save("./phase_retrieval_problem.jld", "solution", U, "KKT_x", solver.program.KKT_x, "s", solver.program.s, "z", solver.program.z, "A", A, "b", b, "c", solver.program.c)

# Uncomment to do warm start from saved result
# data = load("./phase_retrieval_problem.jld")
# cone_qp = warm_start_qp(data, m*n*2)
