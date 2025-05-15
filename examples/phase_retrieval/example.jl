#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using Distributions
using FFTW
using LinearAlgebra
using ToeplitzMatrices
import Random

function get_problem_parameters(A, pinv_A, b, n)
    M = Hermitian(diagm(b) * (I - (A * pinv_A)) * diagm(b))
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

function get_qp(A, pinv_A, b, n)
    A, G, P, b, c, h = get_problem_parameters(A, pinv_A, b, n)
    cones::Vector{Cone} = []
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{Float64, ComplexF64, ComplexF64}(A, G, P, b, c, h, cones)
    return cone_qp
end

# function run_example()
Random.seed!(1)
n = 60

# get the 1-D DFT matrix with n roots of unity
# A = fft(I(Int(n*2)), 1)[:, 1:n]
# A = rand(ComplexF64, (n*2, n*2))[:, :]
m = 2 # number of illumination filters
d = Normal(0, 1) # Normal random variable with mean, variance
A = vcat([Matrix{ComplexF64}(Circulant(rand(d, n))) for i = 1:m]...)
pinv_A = inv(A'*A) * A'

# b = rand(Float64, n)
# create noiseless raw input signal
x = collect(0:0.15:6.6)
ω = 2
b_0 = sin.(ω .* x)[1:end-1]
ϕ = pi/3
b_1 = 0.5 * sin.(2*ω .* x .+ ϕ)[1:end-1]
b = b_0 + b_1
b = append!(b, zeros(length(b)+32))

cone_qp = get_qp(A, pinv_A, b, length(b))
solver = Solver(cone_qp)
solver.max_iterations = 15
status = run_solver(solver)

U = get_solution(solver)
# get the approximate phase vector from the solution
u = eigen(mat(U)).vectors[:, end]
println(eigen(mat(U)).values)
# reconstruct the signal from the phase vector
x = pinv_A * diagm(b) * u
diag_u = diag(mat(U))
println(isapprox(diag_u, ones(length(diag_u))))
println((norm((A * x) - (diagm(b) * u), 2))^2)
println(b - abs.(A * x))

# return status
# end

# run_example()
using CairoMakie
f = Figure()
x_range = 1:1:n
ax = Axis(f[1, 1], title="Input Signal vs Reconstructed Signal over time")
lines!(ax, x_range, b[1:n])
signal = abs.(x)
lines!(ax, x_range, signal[1:n])
display(f)