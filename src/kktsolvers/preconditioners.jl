#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra

function get_kkt_matrix(kktsystem)
    P = kktsystem.P
    A = kktsystem.A
    G = kktsystem.G
    K = [P A' G']
    K_2_2 = zeros(size(A)[1], size(A)[1])
    K_3_3 = zeros(size(G)[1], size(G)[1])
    K = vcat(K, [A K_2_2 zeros(size(A)[1], size(G)[1])])
    K = vcat(K, [G zeros(size(G)[1], size(A)[1]) K_3_3])
    return K
end

function get_identity_preconditioner_matrices(A)
    return I, I
end

function get_jacobi_preconditioner_matrices(A, check=true)
    diag_A = diag(A)
    sqrt_A = sqrt.(diag_A)
    L = diagm(sqrt_A)
    if check == true
        if cond(L) == Inf
            throw(DomainError("Preconditioner matrices are degenerate"))
        end
    end
    inv_M_1 = L
    inv_M_2 = L
    return inv_M_1, inv_M_2
end

function get_spai_preconditioner_matrices(kktsystem, M_0=nothing, m=100)
    """https://tbetcke.github.io/hpc_lecture_notes/it_solvers4.html"""
    A = get_kkt_matrix(kktsystem)
    if isnothing(M_0)
        M_0 = 2 / opnorm(A*A', 1) * A
    end
    M = M_0
    for i=1:m
        C = A * M
        G = I - C
        A_G = A * G
        α = tr(G' * A_G) / norm(A_G, 2)
        M = M + α * G
    end
    inv_L = inv(cholesky(M).L)
    inv_M_1 = inv_L
    inv_M_2 = inv_L
    return inv_M_1, inv_M_2
end

function get_ssor_preconditioner_matrices(kktsystem)
    A = get_kkt_matrix(kktsystem)
    L = LowerTriangular(A)
    D = Diagonal(A)
    U = UpperTriangular(A)
    D_L = D + L
    D_U = D + U
    M = D_L * inv(D) * D_U
    inv_M = inv(M)
    inv_M_1 = inv_M
    inv_M_2 = inv_M
    return inv_M_1, inv_M_2
end

export get_identity_preconditioner_matrices
export get_jacobi_preconditioner_matrices
export get_spai_preconditioner_matrices
export get_ssor_preconditioner_matrices