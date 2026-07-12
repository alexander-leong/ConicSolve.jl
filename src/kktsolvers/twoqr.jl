#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra

function two_qr_solve(device, kktsystem, b_x, b_y, b_z)
    println("two qr 1")
    G = @view kktsystem.G[:, :]
    Q = @view kktsystem.Q[:, :]
    R = @view kktsystem.R[:, :]
    Q_1 = @view Q[:, 1:size(R)[1]]
    Q_2 = @view Q[:, size(R)[1]+1:size(Q)[2]]
    Q_3, R_3 = qr(G * Q_2)
    w = b_z - (G * Q_1 * (R' \ b_y))
    R_3 = [R_3; zeros((size(Q_3, 1)-size(R_3, 1), size(R_3, 2)))]
    u = R_3' \ ((Q_2' * b_x) + (R_3' * (Q_3' * w)))
    W_z = Q_3 * u - w
    y = R \ (Q_1' * b_x - Q_1' * (G' * W_z))
    x = Q_1 * (R' * b_y) + Q_2 * (R_3 \ u)
    z_len = length(b_z)
    z = zeros(Float64, z_len)
    println("two qr 2")
    return [x; y; z]
end

export two_qr_solve