#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("./kktsolvers/conjgrad.jl")
include("./kktsolvers/minres.jl")
include("./kktsolvers/qrchol.jl")

using SparseArrays

mutable struct KKTSystem
    A
    G
    P
    Q
    Q_A
    R
    inv_R
    function KKTSystem(A::AbstractArray{Float64},
                       G::AbstractArray{<:Number},
                       P::AbstractArray{Float64})
        kktmat = new()
        kktmat.A = A
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

    function KKTSystem(G::AbstractArray{<:Number},
                       P::AbstractArray{Float64})
        kktmat = new()
        kktmat.G = G
        kktmat.P = P
        return kktmat
    end
end

function qp_solve(program,
                  G_scaled::AbstractArray{<:Number},
                  inv_W_b_z::AbstractArray{<:Number},
                  solve=full_qr_solve)
    # page 498, 618 Boyd and Vandenberghe
    # page 29, coneprog.pdf
    b_x = @view program.KKT_b[program.inds_c]
    program.kktsystem.G = G_scaled
    x_vec = zeros(Number, length(program.KKT_b))
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

export full_qr_solve