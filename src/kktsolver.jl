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
                       P::AbstractArray{Float64})
        kktmat = new()
        kktmat.G = G
        kktmat.P = P
        if length(size(A)) == 2 && A == isdiag(A)
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

    function KKTSystem(G::AbstractArray{Float64},
                       P::AbstractArray{Float64})
        kktmat = new()
        kktmat.G = G
        kktmat.P = P
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

Calculates the vectors ``x`` and ``y`` (if linear equality constraints present)
by solving the KKT system of linear equations where the KKT matrix is given by kktsystem,
``K`` and b\\_x (i.e. ``b_x``), b\\_z (i.e. ``W^{-T}b_z``) are vectors and
b\\_y (i.e. ``b_y``, if linear equality constraints present).
Setting sparse_G to true may improve performance if G has a particular sparsity structure.
In the absence of linear equalities, the system solved is:
```math
K = \\begin{bmatrix}
    P  & G^T   \\\\
    G  & -W^TW \\\\
\\end{bmatrix}
```
In the presence of linear equalities, the system solved is:
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
\\ y      \\\\
\\end{bmatrix}
```

### Output

The vectors ``x`` and ``y`` (nothing, if not present).
"""
function full_qr_solve(kktsystem, b_x, b_y, b_z, sparse_G=false)
    G = kktsystem.G
    P = kktsystem.P
    if kktsystem.P === nothing
        kktsystem.Q_A = gram(G, sparse_G)
    else
        kktsystem.Q_A = P + gram(G, sparse_G)
    end
    Q_A = kktsystem.Q_A
    b_1 = b_x + G' * b_z
    b_2 = b_1
    if !isdefined(kktsystem, :Q) && !isdefined(kktsystem, :R)
        # no linear equality constraints
        x = Q_A \ b_1
        return x, nothing
    else
        Q = kktsystem.Q
        R = kktsystem.R
        Q_1_x = kktsystem.inv_R' * b_y
        Q_1 = @view Q[:, 1:length(Q_1_x)]
        Q_2 = @view Q[:, length(Q_1_x)+1:size(Q)[2]]
        Q_1_A = Q_1' * Q_A
        Q_21_A = Q_2' * Q_A
        Q_12_A = Q_21_A'
        U = cholesky(Q_A).U * Q_2
        Q_2_A = cholesky(U' * U)
        A₁₁x₁ = Q_1_A * Q_1 * Q_1_x
        A₂₁x₁ = Q_21_A * Q_1 * Q_1_x
        inv_Q_2_A = inv(Q_2_A.U)
        L = inv_Q_2_A * inv_Q_2_A'
        Q_2_x = L * (Q_2' * b_2 - A₂₁x₁)
        A₁₂x₂ = Q_1' * Q_12_A * Q_2_x
        y = R \ ((Q_1' * b_1) - A₁₁x₁ - A₁₂x₂)
        x = Q * [Q_1_x; Q_2_x]
        return x, y
    end
end

function qp_solve(program,
                  G_scaled::AbstractArray{Float64},
                  inv_W_b_z::AbstractArray{Float64},
                  solve=full_qr_solve)
    # page 498, 618 Boyd and Vandenberghe
    # page 29, coneprog.pdf
    b_x = @view program.KKT_b[program.inds_c]
    program.kktsystem.G = G_scaled
    x_vec = zeros(Float64, length(program.KKT_b))
    if isdefined(program.kktsystem, :Q) && isdefined(program.kktsystem, :R)
        b_y = @view program.KKT_b[program.inds_b]
        x, y = solve(program.kktsystem, b_x, b_y, inv_W_b_z)
        x_vec[program.inds_b] = y
    else
        b_y = nothing
        x, y = solve(program.kktsystem, b_x, b_y, inv_W_b_z)
    end
    x_vec[program.inds_c] = x
    return x_vec
end

export KKTSystem
export full_qr_solve