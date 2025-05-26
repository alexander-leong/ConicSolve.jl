#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""This file contains functions for solving the KKT system"""

include("./cuda/qrchol.jl")

function sparse_gram(G)
    spG = sparse(G)
    return spG' * spG
end

function gram(G, sparse_G=false)
    if sparse_G == true
        return sparse_gram(G)
    end
    R = qr(G).R
    return R' * R
end

"""
    full_qr_solve(kktsystem, kkt_1_1, b_x, b_y, b_z)

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
function full_qr_solve(kktsystem, kkt_1_1, b_x, b_y, b_z, check=false)
    G = kktsystem.G
    kktsystem.Q_A = kkt_1_1 + 1e-3*I
    Q_A = kktsystem.Q_A
    b_1 = b_x + G' * b_z
    b_2 = b_1
    x_len = size(kktsystem.G)[2]
    if isdefined(kktsystem, :A)
        y_len = size(kktsystem.A)[1]
    else
        y_len = 0
    end
    z_len = size(kktsystem.G)[1]
    n = x_len + y_len + z_len
    # x_vec = zeros(Float64, n)
    if !isdefined(kktsystem, :Q) && !isdefined(kktsystem, :R)
        # no linear equality constraints
        x = Q_A \ b_1
        # x_vec[1:x_len] = x
        # return x_vec
    else
        Q = kktsystem.Q
        R = kktsystem.R
        Q_1 = @view Q[:, 1:size(R)[1]]
        Q_2 = @view Q[:, size(R)[1]+1:size(Q)[2]]
        Q, Q_A, Q_1, Q_2, R, b_y, b_1, b_2 = qr_chol_cpu_to_gpu(Q, Q_A, Q_1, Q_2, R, b_y, b_1, b_2)
        Q_1_x = R' \ b_y
        solve_elapsed_time = @elapsed begin
        U = cholesky(Q_A, check=check).U * Q_2
        Q_2_A = cholesky(U' * U, check=check).L
        end
        println(solve_elapsed_time)
        S_Q1_x = Q_A * (Q_1 * Q_1_x)
        A₁₁x₁ = Q_1' * S_Q1_x
        A₂₁x₁ = Q_2' * S_Q1_x
        U_Q2_x = Q_2_A \ (Q_2' * b_2 - A₂₁x₁)
        Q_2_x = Q_2_A' \ U_Q2_x
        A₁₂x₂ = Q_1' * (Q_A' * (Q_2 * Q_2_x))
        y = R \ ((Q_1' * b_1) - A₁₁x₁ - A₁₂x₂)
        x = Q * [Q_1_x; Q_2_x]
        x_cpu = zeros(Float64, x_len)
        y_cpu = zeros(Float64, y_len)
        z_cpu = zeros(Float64, z_len)
        qr_chol_gpu_to_cpu(x_cpu, x, y_cpu, y)
        return [x_cpu; y_cpu; z_cpu]
    end
end

export KKTSystem
export full_qr_solve