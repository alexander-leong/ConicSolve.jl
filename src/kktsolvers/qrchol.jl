#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""This file contains functions for solving the KKT system"""

include("./cuda/qrchol.jl")

"""
    qr_chol_solve(device, kktsystem, b_x, b_y, b_z, check)

Calculates the vectors ``x`` and ``y`` (if linear equality constraints present)
by solving the KKT system of linear equations where the KKT matrix is given by kktsystem,
``K`` and b\\_x (i.e. ``b_x``), b\\_z (i.e. ``W^{-T}b_z``) are vectors and
b\\_y (i.e. ``b_y``, if linear equality constraints present).

### Parameters

- `device`: `CPU` or `GPU`
- `kktsystem`: KKTSystem object
- `b_x`: The x component of the vector b
- `b_y`: The y component of the vector b
- `b_z`: The z component of the vector b
- `check`: true or false depending on whether matrix decompositions are valid

### Output

The KKT solution vector [x; y; 0].
"""
function qr_chol_solve(device, kktsystem, b_x, b_y, b_z, check=false)
    G = @view kktsystem.G[:, :]
    Q_A = @view kktsystem.kkt_1_1[:, :]
    Q_A += 1e-3*I
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
    if !isdefined(kktsystem, :Q) && !isdefined(kktsystem, :R)
        # no linear equality constraints
        x = Q_A \ b_1
        z = zeros(Float64, z_len)
        return [x; z]
    else
        Q = @view kktsystem.Q[:, :]
        R = @view kktsystem.R[:, :]
        Q_1 = @view Q[:, 1:size(R)[1]]
        Q_2 = @view Q[:, size(R)[1]+1:size(Q)[2]]
        Q_A = get_array(device, Q_A)
        Q_2 = get_array(device, Q_2)
        Q_A = cholesky(Q_A, check=check).U * Q_2
        R = get_array(device, R)
        b_y = get_array(device, b_y)
        Q_1 = get_array(device, Q_1)
        Q_2_A = cholesky(Q_A' * Q_A, check=check).L
        Q_A = @view kktsystem.kkt_1_1[:, :]
        Q_A = get_array(device, Q_A)
	    Q_1_x = R' \ b_y
        S_Q1_x = Q_A * (Q_1 * Q_1_x)
        A₁₁x₁ = Q_1' * S_Q1_x
        A₂₁x₁ = Q_2' * S_Q1_x
        b_2 = get_array(device, b_2)
        U_Q2_x = Q_2_A \ (Q_2' * b_2 - A₂₁x₁)
        Q_2_x = Q_2_A' \ U_Q2_x
        A₁₂x₂ = Q_1' * (Q_A' * (Q_2 * Q_2_x))
        b_1 = get_array(device, b_1)
        y = R \ ((Q_1' * b_1) - A₁₁x₁ - A₁₂x₂)
        Q = get_array(device, Q)
        x = Q * [Q_1_x; Q_2_x]
        if device == GPU
            x_cpu = zeros(Float64, x_len)
            y_cpu = zeros(Float64, y_len)
            z_cpu = zeros(Float64, z_len)
            copyto!(x_cpu, x)
            copyto!(y_cpu, y)
            return [x_cpu; y_cpu; z_cpu]
        else
            z = zeros(Float64, z_len)
            return [x; y; z]
        end
    end
end

export qr_chol_solve
