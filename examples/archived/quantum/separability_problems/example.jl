#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using LinearAlgebra

function partial_transpose(ρ, dims, sys=2)
    if sys == 1
        return ρ'
    end
    (m, n) = dims
    ρ_T = zeros(ComplexF64, size(ρ))
    for i = 1:Int(size(ρ)[1]/m)
        for j = 1:Int(size(ρ)[2]/n)
            is = m*(i-1)+1:m*(i)
            js = n*(j-1)+1:n*(j)
            ρ_T[is, js] = ρ[is, js]'
        end
    end
    return ρ_T
end

function get_problem_parameters(ρ)
    # for 2⊗2 Hilbert space
    # basis must satisfy eq. 15
    σ_1 = [1 0; 0 1] ./ 2
    σ_2 = [0 1; 1 0] ./ 2
    σ_3 = [0 -im; im 0] ./ 2
    σ_4 = [1 0; 0 -1] ./ 2
    σ_vec = []
    push!(σ_vec, σ_1)
    push!(σ_vec, σ_2)
    push!(σ_vec, σ_3)
    push!(σ_vec, σ_4)
    α = 2
    ρ_i_j_vec = []
    n_dims = 2
    for i = 1:n_dims^2
        for j = 1:n_dims^2
            ρ_i_j = 1 / α^2 * tr(ρ * kron(σ_vec[i], σ_vec[j]))
            push!(ρ_i_j_vec, ρ_i_j)
        end
    end
    f_σ_j = σ_j -> kron(ρ_i_j_vec[σ_j[1]] * σ_1, σ_j[2], σ_1)
    G_0 = reduce(+, map(f_σ_j, enumerate(σ_vec)))
    for i = 2:n_dims^2
        for j = 1:n_dims^2
            k_0 = kron(σ_vec[i], σ_vec[j], σ_vec[1])
            k_1 = kron(σ_vec[1], σ_vec[j], σ_vec[i])
            k = k_0 + k_1
            G_0 += ρ_i_j_vec[n_dims*(i-1)+j] * k
        end
    end
    partial_G_A = svec(partial_transpose(G_0, (2, 2), 1))
    partial_G_B = svec(partial_transpose(G_0, (2, 2), 2))
    G_0 = svec(G_0)
    G = G_0
    for i = 2:n_dims^2
        for j = 1:n_dims^2
            G_iji = kron(σ_vec[i], σ_vec[j], σ_vec[i])
            partial_G_A_iji = partial_transpose(G_iji, (2, 2), 1)
            partial_G_B_iji = partial_transpose(G_iji, (2, 2), 2)
            G = hcat(G, svec(G_iji))
            partial_G_A = hcat(partial_G_A, svec(partial_G_A_iji))
            partial_G_B = hcat(partial_G_B, svec(partial_G_B_iji))
        end
    end
    for i = 2:n_dims^2
        for j = 1:n_dims^2
            for k = i+1:n_dims^2
                k_0 = kron(σ_vec[i], σ_vec[j], σ_vec[k])
                k_1 = kron(σ_vec[k], σ_vec[j], σ_vec[i])
                G_ijk = k_0 + k_1
                partial_G_A_ijk = partial_transpose(G_ijk, (2, 2), 1)
                partial_G_B_ijk = partial_transpose(G_ijk, (2, 2), 2)
                G = hcat(G, svec(G_ijk))
                partial_G_A = hcat(partial_G_A, svec(partial_G_A_ijk))
                partial_G_B = hcat(partial_G_B, svec(partial_G_B_ijk))
            end
        end
    end
    G = vcat(G, partial_G_A, partial_G_B)
    G = -G
    n = size(G)[2]
    P = zeros((n, n))
    A = zeros((1, n))
    A[1, 1] = 1
    b = [1.0]
    c = zeros(n)
    h = zeros(size(G)[1])
    return A, G, P, b, c, h
end

function get_qp()
    # the separable case
    ρ = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
    A, G, P, b, c, h = get_problem_parameters(ρ)
    cones::Vector{Cone} = []
    n = (2^2)*2
    push!(cones, PSDCone(n))
    push!(cones, PSDCone(n))
    push!(cones, PSDCone(n))
    cone_qp = ConeQP{ComplexF64}(A, G, P, b, c, h, cones)
    return cone_qp
end

function run_example()
    cone_qp = get_qp()
    solver = Solver(cone_qp)
    status = optimize!(solver)
    return status
end

run_example()
