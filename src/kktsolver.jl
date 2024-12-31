#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""This file contains functions for solving the KKT system"""

using SparseArrays

mutable struct KKTSystem
    G
    P
    Q
    Q_A
    R
    inv_R
    function KKTSystem(A::AbstractArray{Float64},
                       G::AbstractArray{Float64},
                       P::AbstractArray{Float64}=nothing)
        kktmat = new()
        kktmat.G = G
        kktmat.P = P
        if A == isdiag(A)
            kktmat.Q, kktmat.R = Matrix(I, size(A)[1], size(A)[2]), A
        else
            # this solver uses the Julia (LAPACK based) QR factorization method
            # which is based on Householder reflections.
            Q, kktmat.R = qr(A')
            kktmat.Q = collect(Q)
        end
        kktmat.inv_R = inv(kktmat.R)
        return kktmat
    end
end

function sparse_gram(G)
    spG = sparse(G)
    return spG' * spG
end

function gram(G, sparse_G=false)
    if sparse_G == true
        return sparse_gram(G)
    end
    return G' * G
end

"""
    full_qr_solve(kktsystem, b_x, b_y, b_z, sparse_G=false)

Calculates the vectors ``x`` and ``y`` by solving the KKT system of linear equations where the KKT matrix is given
by kktsystem, ``K`` and b\\_x (i.e. ``b_x``), b\\_y (i.e. ``b_y``), b\\_z (i.e. ``W^{-T}b_z``) are vectors.
Setting sparse_G to true may improve performance if G has a particular sparsity structure.
```math
K = \\begin{bmatrix}
    P  & G^T   & A^T \\\\
    G  & -W^TW & 0   \\\\
    A  & 0     & 0
\\end{bmatrix}
\\hspace{1cm}
x = \\begin{bmatrix}
\\ Q_1^Tx \\\\
\\ Q_2^Tx \\\\
\\end{bmatrix}
```

### Output

The vectors ``x`` and ``y``.
"""
function full_qr_solve(kktsystem, b_x, b_y, b_z, sparse_G=false)
    G = kktsystem.G
    P = kktsystem.P
    Q = kktsystem.Q
    R = kktsystem.R
    if kktsystem.P === nothing
        kktsystem.Q_A = gram(G, sparse_G)
    else
        kktsystem.Q_A = P + gram(G, sparse_G)
    end
    Q_A = kktsystem.Q_A
    # solve for Q_1_x from third equation
    Q_1_x = kktsystem.inv_R' * b_y
    Q_1 = @view Q[:, 1:length(Q_1_x)]
    Q_2 = @view Q[:, length(Q_1_x)+1:size(Q)[2]]
    b_1 = b_x + G' * b_z
    b_2 = b_1
    Q_1_A = Q_1' * Q_A
    Q_21_A = Q_2' * Q_A
    # solve for Q_2_x from second equation
    Q_12_A = Q_21_A'
    Q_2_A = cholesky(Q_2' * Q_A * Q_2)
    A₁₁x₁ = Q_1_A * Q_1 * Q_1_x
    A₂₁x₁ = Q_21_A * Q_1 * Q_1_x
    inv_Q_2_A = inv(Q_2_A.U)
    L = inv_Q_2_A * inv_Q_2_A'
    Q_2_x = L * (Q_2' * b_2 - A₂₁x₁)
    # solve for y from first equation
    A₁₂x₂ = Q_1' * Q_12_A * Q_2_x
    y = R \ ((Q_1' * b_1) - A₁₁x₁ - A₁₂x₂)
    x = Q * [Q_1_x; Q_2_x]
    return x, y
end

function reduced_qr_solve(A, G, P, b_x, b_y, b_z, inv_W, program)
    # do full QR, eq. 55, coneprog.pdf
    Q, R = qr(A')
    R₁ = R
    Q = collect(Q)
    Q₁ = Q[:, 1:size(R)[2]]
    Q₂ = Q[:, size(R)[2]+1:size(Q)[2]]
    # do cholesky, eq. 55, coneprog.pdf
    Q_A = P + G' * G
    Q_A_QR = qr(Q_A)
    Q_A_QR_sqrt = Q₂' * Q_A_QR.R'
    L = cholesky(Q_A_QR_sqrt * Q_A_QR_sqrt').L
    Q₁ᵀ_x = inv(R₁)' * b_y
    # solve for x
    x = Q₁ * Q₁ᵀ_x
    b_2 = (b_x + G' * inv_W' * b_z)
    A₂₁ = Q₂' * Q_A * Q₁
    Q₂ᵀ_x = (L * L') \ ((Q₂' * b_2) - (A₂₁ * Q₁ᵀ_x))
    b_1 = b_2
    A₁ = Q₁' * Q_A
    # solve for y
    A₁₁x₁ = A₁ * Q₁ * Q₁ᵀ_x
    A₁₂x₂ = A₁ * Q₂ * Q₂ᵀ_x
    y = R \ ((Q₁' * b_1) - A₁₁x₁ - A₁₂x₂)
    return x, y
end

function qp_solve(program,
                  G_scaled::AbstractArray{Float64},
                  inv_W_b_z::AbstractArray{Float64},
                  solve=full_qr_solve)
    # page 498, 618 Boyd and Vandenberghe
    # page 29, coneprog.pdf
    b_x = @view program.KKT_b[program.inds_c]
    b_y = @view program.KKT_b[program.inds_b]
    program.kktsystem.G = G_scaled
    x, y = solve(program.kktsystem, b_x, b_y, inv_W_b_z)
    x_vec = zeros(length(program.KKT_b))
    x_vec[program.inds_c] = x
    x_vec[program.inds_b] = y
    return x_vec
end

export KKTSystem
export full_qr_solve