#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
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
    println(length(c))
    println(size(A))
    h = zeros(size(G)[1])
    cones::Vector{Cone} = []
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    s = data["s"]
    z = data["z"]
    cone_qp.s = s
    cone_qp.z = z
    cone_qp.KKT_x = data["KKT_x"]
    return cone_qp
end

function get_qp(A, pinv_A, b, n)
    A, G, P, b, c, h = get_problem_parameters(A, pinv_A, b, n)
    save("/home/alexander/Documents/alexander_leong/ConicSolve.jl/phase_retrieval_params.jld", "A", A, "G", G, "P", P, "b", b, "c", c, "h", h)
    cones::Vector{Cone} = []
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end

# function run_example()
Random.seed!(1)

# set parameters
m = 4 # number of illumination filters
n = 33
x = collect(0:0.2:6.4)

# get the 1-D DFT matrix with n roots of unity
F = fft(I(Int(n)), 1)
# F_Re = real(F)
# F_Im = imag(F)
d = Normal(0, 0.01) # Normal random variable with mean, variance
# A_Re = vcat([Matrix{Float64}(Circulant(F_Re * rand(d, n))) for i = 1:m]...)
# A_Im = vcat([Matrix{Float64}(Circulant(F_Im * rand(d, n))) for i = 1:m]...)
# A = A_Re + A_Im
A = vcat([Matrix{ComplexF32}(Circulant(F * rand(d, n))) for i = 1:m]...)

# generate original signal
# create noiseless raw input signal
ω = 1
ϕ = pi/3
x_sig = sin.(ω .* x)[1:end] # .+ (0.5 * sin.(2*ω .* x .+ ϕ)[1:end])
# calculate measurements by passing original signal through each filter
b = abs.(A * x_sig)
# println(size(A))
# println(size(b))

# A = [A_Re -A_Im; A_Im A_Re]
pinv_A = inv(A'*A) * A'
# pinv_A = pinv(A)

# free memory
F = 0
F_Re = 0
F_Im = 0
A_Re = 0
A_Im = 0

# b = rand(Float64, n)
# b_0 = sin.(ω .* x)[1:end]
# b_1 = 0.5 * sin.(2*ω .* x .+ ϕ)[1:end]
# b = b_0 + b_1
# b = repeat(b, m*2)
# b = repeat(b, 2)

# data = load("/home/alexander/Documents/alexander_leong/ConicSolve.jl/phase_retrieval_4_30_simple.jld")
# b = ones(m*n*2)
# c = data["c"]
cone_qp = get_qp(A, pinv_A, b, length(b)*2)
# cone_qp = warm_start_qp(data, m*n*2)
# cone_qp.b = b
# cone_qp.c = c
solver = Solver(cone_qp)
solver.max_iterations = 5
solver.num_threads = 1
status = run_solver(solver)

U = get_solution(solver)
# U = mat(U)
# U = U + (I / sqrt(2))
# save result
# println(length(solver.program.KKT_x))

A = data["A"]
b = data["b"]
save("/home/alexander/Documents/alexander_leong/ConicSolve.jl/phase_retrieval_4_30_simple.jld", "solution", U, "KKT_x", solver.program.KKT_x, "s", solver.program.s, "z", solver.program.z, "A", A, "b", b, "c", solver.program.c)
# get the approximate phase vector from the solution
# X = U[1:m*n, 1:m*n] + (U[m*n+1:end, 1:m*n] * im)
# s, v = eigen(X)
# u = v[:, end]
# println(s)
# reconstruct the signal from the phase vector
# x = pinv_A * diagm(b) * u
# diag_u = diag(mat(U))
# println(isapprox(diag_u, ones(length(diag_u))))
# println((norm((A * x) - (diagm(b) * u), 2))^2)
# println(b - abs.(A * x))

# return status
# end

# run_example()
# using CairoMakie
# f = Figure()
# x_range = 1:1:n
# ax = Axis(f[1, 1], title="Input Signal vs Reconstructed Signal over time")
# lines!(ax, x_range, b[1:n])
# signal = abs.(x)
# lines!(ax, x_range, signal[1:n])
# display(f)
