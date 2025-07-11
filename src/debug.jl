#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""
This function contains functions only used for debugging
the correctness of the optimization solver (not to be used in production).
"""

using LinearAlgebra

function is_convex_cone(program, α, etol=1e-3)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        s_hat = program.s[inds]
        z_hat = program.z[inds]
        is_convex_cone(cone, α, s_hat, z_hat, etol)
    end
    return true
end

function is_convex_cone(cone::NonNegativeOrthant, α, s_hat, z_hat, etol=1e-3)
    if cone.z isa AbstractArray{ComplexF64}
        s_result = map(x -> minimum((real(x), imag(x))), s_hat + α * cone.s)
        @assert minimum(s_result) >= -etol
        z_result = map(x -> minimum((real(x), imag(x))), z_hat + α * cone.z)
        @assert minimum(z_result) >= -etol
    else
        @assert minimum(s_hat + α * cone.s) >= -etol
        @assert minimum(z_hat + α * cone.z) >= -etol
    end
end

function is_convex_cone(cone::PSDCone, α, s_hat, z_hat, etol=1e-3)
    @assert minimum(eigen(mat((s_hat + α * cone.s))).values) >= -etol
    @assert minimum(eigen(mat((z_hat + α * cone.z))).values) >= -etol
end

function is_convex_cone(cone::SecondOrderCone, α, s_hat, z_hat, etol=1e-3)
    s_hat_new = s_hat + α * cone.s
    z_hat_new = z_hat + α * cone.z
    @assert norm(s_hat_new[2:end]) - etol <= s_hat_new[1]
    @assert norm(z_hat_new[2:end]) - etol <= z_hat_new[1]
end

function check_linear_equalities(program,
                                 r::Vector{T},
                                 tol::Float64=1e-3) where T <: Number
    y_inds = program.inds_b
    z_inds = program.inds_h
    b_y = -r[y_inds]
    b_z = -r[z_inds]
    b_y_is_optimal = all(x->abs(x) <= tol, b_y)
    b_z_is_optimal = all(x->abs(x) <= tol, b_z)
    return b_y_is_optimal && b_z_is_optimal
end

function check_linear_inequalities(s::Vector{T},
                                   z::AbstractArray,
                                   tol::Float64=1e-3) where T <: Number
    r2 = all(s->norm(s) >= tol, s) && all(z->norm(z) >= tol, z)
    return r2
end

function check_constraints(program,
                           r,
                           s,
                           z,
                           tol,
                           check_r1=true,
                           check_r2=true,
                           check_r3=true)
    x_inds = program.inds_c
    b_x = r[x_inds]
    b_x_is_optimal = all(x->abs(x) <= tol, b_x)
    x = program.KKT_x
    
    r1 = check_r1 ? b_x_is_optimal : true
    check_r2 = check_r2 && isdefined(program, :A)
    r2 = check_r2 ? check_linear_equalities(program, r) : true
    r3 = check_r3 ? check_linear_inequalities(s, z) : true

    result = r1 && r2 && r3
    return result
end

function check_affine_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled, etol=1e-3)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        λ_k = λ[inds]
        circ_λ_k = circ(cone, -λ_k, λ_k)
        @assert isapprox(circ(cone, λ_k, Δzₐ_scaled[inds] + Δsₐ_scaled[inds]), circ_λ_k, rtol=etol)
    end
    return true
end

function check_combined_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled, Δs_scaled, Δz_scaled, μ, σ, etol=1e-3)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        λ_k = λ[inds]
        e = get_e(cone)
        circ_λ_k = circ(cone, -λ_k, λ_k)
        d_s = circ_λ_k - circ(cone, Δsₐ_scaled[inds], Δzₐ_scaled[inds]) + σ * μ * e
        @assert isapprox(circ(cone, λ_k, (Δz_scaled + Δs_scaled)[inds]), d_s, rtol=etol)
    end
    return true
end