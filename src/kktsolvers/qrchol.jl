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
    G = @view kktsystem.G[:, :]
    @views kkt_1_1 += 1e-3*I
    Q_A = @view kkt_1_1[:, :]
    b_1 = b_x + G' * b_z
    b_2 = @view b_1[:]
    x_len = length(b_x)
    if isdefined(kktsystem, :A)
    	A = @view kktsystem.A[:, :]
        y_len = size(A)[1]
    else
        y_len = 0
    end
    z_len = length(b_z)
    n = x_len + y_len + z_len
    # x_vec = zeros(Float64, n)
    function cu_cholesky(A, uplo)
        if uplo == 'L'
            result = tril(CUSOLVER.Xpotrf!(uplo, A)[1])
        else
            result = triu(CUSOLVER.Xpotrf!(uplo, A)[1])
        end
        return result
    end
    if !isdefined(kktsystem, :Q) && !isdefined(kktsystem, :R)
        # no linear equality constraints
        x = Q_A \ b_1
        # x_vec[1:x_len] = x
        # return x_vec
    else
        Q = @view kktsystem.Q[:, :]
        R = @view kktsystem.R[:, :]
        Q_1 = @view Q[:, 1:size(R)[1]]
        Q_2 = @view Q[:, size(R)[1]+1:size(Q)[2]]
        Q_A = CuArray{Float32}(Q_A)
        Q_2 = CuArray{Float32}(Q_2)
        solve_elapsed_time = @elapsed begin
        Q_A = cholesky(Q_A, check=check).U * Q_2
        R = CuArray{Float32}(R)
        b_y = CuArray{Float32}(b_y)
        Q_1 = CuArray{Float32}(Q_1)
        Q_2_A = cholesky(Q_A' * Q_A, check=check).L
        Q_A = @view kkt_1_1[:, :]
        Q_A = CuArray{Float32}(Q_A)
	    Q_1_x = R' \ b_y
        S_Q1_x = Q_A * (Q_1 * Q_1_x)
        A₁₁x₁ = Q_1' * S_Q1_x
        A₂₁x₁ = Q_2' * S_Q1_x
        b_2 = CuArray{Float32}(b_2)
        U_Q2_x = Q_2_A \ (Q_2' * b_2 - A₂₁x₁)
        Q_2_x = Q_2_A' \ U_Q2_x
        A₁₂x₂ = Q_1' * (Q_A' * (Q_2 * Q_2_x))
        b_1 = CuArray{Float32}(b_1)
        y = R \ ((Q_1' * b_1) - A₁₁x₁ - A₁₂x₂)
        Q = CuArray{Float32}(Q)
        x = Q * [Q_1_x; Q_2_x]
        end
        println(solve_elapsed_time)
        x_cpu = zeros(Float64, x_len)
        y_cpu = zeros(Float64, y_len)
        z_cpu = zeros(Float64, z_len)
        copyto!(x_cpu, x)
        copyto!(y_cpu, y)
        return [x_cpu; y_cpu; z_cpu]
    end
end

export KKTSystem
export full_qr_solve
