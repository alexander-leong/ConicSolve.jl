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

function get_problem_parameters(A, pinv_A, b, n)
    b = vec([b b])
    M = Hermitian(diagm(b) * (I - (A * pinv_A)) * diagm(b))
    idx = map(x -> CartesianIndex(x, x), 1:n)
    idx = lower_triangular_from_2d_idx(n, idx)
    idx = map(x -> CartesianIndex(x[1], x[2]), enumerate(idx))
    A = zeros((n, Int(n * (n + 1) / 2)))
    A[idx] .= 1
    b = ones(length(b))
    G = -1.0I(size(A)[2])
    P = zeros((size(A)[2], size(A)[2]))
    c = svec(M)
    h = zeros(size(G)[1])
    return A, G, nothing, b, c, h
end

function get_qp(A, pinv_A, b, n)
    A, G, P, b, c, h = get_problem_parameters(A, pinv_A, b, n)
    cones::Vector{Cone} = []
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end

# function run_example()
Random.seed!(1)
n = 41

# get the 1-D DFT matrix with n roots of unity
F = fft(I(Int(n)), 1)
F_Re = real(F)
F_Im = imag(F)
m = 2 # number of illumination filters
d = Normal(0, 1) # Normal random variable with mean, variance
A_Re = vcat([Matrix{Float64}(Circulant(F_Re * rand(d, n))) for i = 1:m]...)
A_Im = vcat([Matrix{Float64}(Circulant(F_Im * rand(d, n))) for i = 1:m]...)
A = [A_Re -A_Im; A_Im A_Re]
pinv_A = inv(A'*A) * A'

# free memory
F_Re = 0
F_Im = 0
A_Re = 0
A_Im = 0

# b = rand(Float64, n)
# create noiseless raw input signal
x = collect(0:0.16:6.4)
ω = 2
b_0 = sin.(ω .* x)[1:end]
ϕ = pi/3
b_1 = 0.5 * sin.(2*ω .* x .+ ϕ)[1:end]
b = b_0 + b_1
b = append!(b, zeros(length(b)))

cone_qp = get_qp(A, pinv_A, b, 2*length(b))
solver = Solver(cone_qp)
solver.max_iterations = 15
solver.num_threads = 8
status = run_solver(solver)

U = get_solution(solver)
U = mat(U)
U = U + (I / sqrt(2))
X = U[1:m*n, 1:m*n] + (U[m*n+1:end, 1:m*n] * im)
# save result
save("/home/alexander/Documents/alexander_leong/ConicSolve.jl/phase_retrieval.jld", "solution", X, "s", solver.program.s)
# get the approximate phase vector from the solution
s, v = eigen(X)
u = v[:, end]
println(s)
# reconstruct the signal from the phase vector
# x = pinv_A * diagm(b) * u
# diag_u = diag(mat(U))
# println(isapprox(diag_u, ones(length(diag_u))))
# println((norm((A * x) - (diagm(b) * u), 2))^2)
# println(b - abs.(A * x))

# return status
# end

# run_example()
using CairoMakie
f = Figure()
x_range = 1:1:n
ax = Axis(f[1, 1], title="Input Signal vs Reconstructed Signal over time")
lines!(ax, x_range, b[1:n])
signal = abs.(x)
# lines!(ax, x_range, signal[1:n])
# display(f)