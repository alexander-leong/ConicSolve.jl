#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("conjgrad.jl")
include("minres.jl")
include("preconditioners.jl")
include("qrchol.jl")

using LinearAlgebra
using Logging

function get_preconditioners()
    preconditioners = Dict(
        "jacobi" => get_jacobi_preconditioner_matrices,
        "none" => get_identity_preconditioner_matrices,
        "spai" => get_spai_preconditioner_matrices,
        "ssor" => get_ssor_preconditioner_matrices
    )
    return preconditioners
end

function get_kkt_solvers()
    kkt_solvers = Dict(
        "conjgrad" => Dict("label" => "Conjugate Gradient",
                           "fn" => conj_grad_kkt_solve,
                           "iterative" => true),
        "minres" => Dict("label" => "MINRES",
                         "fn" => minres_kkt_solve,
                         "iterative" => true),
        "qrchol" => Dict("label" => "QR and Cholesky",
                         "fn" => full_qr_solve,
                         "iterative" => false)
    )
    return kkt_solvers
end

function kktmatmul(kktsystem, kkt_1_1, x)
    P = kktsystem.P
    A = kktsystem.A
    inds_c = 1:size(P)[1]
    inds_b = inds_c[end]+1:inds_c[end]+size(A)[1]
    # inds_h = inds_b[end]+1:inds_b[end]+size(G)[1]
    # P_x = P * x[inds_c]
    P_x = kkt_1_1 * x[inds_c]
    A_x = A' * x[inds_b]
    # G_x = G' * x[inds_h]
    # result = vcat(P_x + A_x + G_x,
    #     A * x[inds_c],
    #     G * x[inds_c] - x[inds_h])
    result = vcat(P_x + A_x,
        A * x[inds_c])
    return result
end

function get_residual(kktsystem, kkt_1_1, b, x)
    result = kktmatmul(kktsystem, kkt_1_1, x)
    r = b - result
    return r
end

export get_kkt_solvers
export get_preconditioners
export get_residual
export kktmatmul
