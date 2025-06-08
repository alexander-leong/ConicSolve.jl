#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("./kktsolvers/itersolver.jl")

using SparseArrays

mutable struct KKTSystem
    A
    G
    P
    Q
    Q_A
    R
    # inv_R
    kkt_1_1
    function KKTSystem(A::AbstractArray{Float64},
                       G::AbstractArray{<:Number},
                       P::Union{AbstractArray{Float64}, Nothing})
        kktmat = new()
        kktmat.A = @view A[:, :]
        kktmat.G = @view G[:, :]
        kktmat.P = P
        if length(size(A)) == 2 && A == isdiag(A)
            kktmat.Q, kktmat.R = Matrix(I, size(A)[1], size(A)[2]), A
        else
            # this solver uses the Julia (LAPACK based) QR factorization method
            # which is based on Householder reflections.
            Q, kktmat.R = qr(A')
            kktmat.Q = collect(Q)
        end
        # kktmat.inv_R = inv(kktmat.R)
        return kktmat
    end

    function KKTSystem(G::AbstractArray{<:Number},
                       P::Union{AbstractArray{Float64}, Nothing})
        kktmat = new()
        kktmat.G = G
        kktmat.P = P
        return kktmat
    end
end

mutable struct KKTSolver
    affine_search_direction
    combined_search_direction
    kktsolve
    preconditioner::String
    preconditioners::Dict
end

function setup_default_kkt_solver(kktsolve,
                                  preconditioner="none",
                                  preconditioners=nothing)
    if isnothing(preconditioners)
        preconditioners = get_preconditioners()
    end
    kktsolvers = get_kkt_solvers()
    kktsolve_var = kktsolvers[kktsolve]
    iterative = kktsolve_var["iterative"]
    if iterative
        solver = KKTSolver(get_iterative_affine_search_direction,
                           get_iterative_combined_search_direction,
                           kktsolve_var,
                           preconditioner,
                           preconditioners)
        return solver
    end
    solver = KKTSolver(get_affine_search_direction,
                       get_combined_search_direction,
                       kktsolve_var,
                       preconditioner,
                       preconditioners)
    return solver
end

function get_solve_args(program)
    if isdefined(program.kktsystem, :Q)
        if isdefined(program.kktsystem, :R)
            b_y = @view program.KKT_b[program.inds_b]
            return b_y
        end
    end
    b_y = nothing
    return b_y
end

function qp_solve(solver,
                  G_scaled::AbstractArray{<:Number},
                  inv_W_b_z::AbstractArray{<:Number},
                  solve=full_qr_solve)
    # page 498, 618 Boyd and Vandenberghe
    # page 29, coneprog.pdf
    program = solver.program
    b_x = @view program.KKT_b[program.inds_c]
    program.kktsystem.G = G_scaled
    b_y = get_solve_args(program)
    device = solver.device
    x_vec = solve(device, program.kktsystem, program.kktsystem.kkt_1_1, b_x, b_y, inv_W_b_z)
    return x_vec
end

function qp_solve_iterative(solver,
                            G_scaled::AbstractArray{<:Number},
                            kkt_1_1,
                            inv_W_b_z::AbstractArray{<:Number},
                            solve=minres_kkt_solve)
    program = solver.program
    KKT_b_x = @view program.KKT_b[program.inds_c]
    b_x = KKT_b_x + G_scaled' * inv_W_b_z
    b = vcat(b_x, program.KKT_b[program.inds_b])
    b = b[:, 1]
    kktsystem = program.kktsystem
    kktsolver = solver.kktsolver
    preconditioner = kktsolver.preconditioner
    preconditioner_fn = kktsolver.preconditioners[preconditioner]
    # b_x = @view b[program.inds_c]
    # FIXME b_y = get_solve_args(program)
    # b = vcat(b_x, b_y, inv_W_b_z)
    # b = vcat(b_x, b_y)
    x_0 = @view program.KKT_x[1:length(b)]
    kktsystem.G = G_scaled
    # FIXME
    inv_M_1, inv_M_2 = preconditioner_fn(kktsystem)
    device = solver.device
    x_vec = solve(device, kktsystem, kkt_1_1, b, x_0, inv_M_1, inv_M_2)
    return x_vec
end
