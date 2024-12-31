#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""
This function contains functions only used for debugging
the correctness of the optimization solver (not to be used in production).
"""

include("./cones/nonneg.jl")
include("./cones/psdcone.jl")

using LinearAlgebra

function is_convex_cone(program, α, etol=1e-6)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        s_hat = program.s[inds]
        z_hat = program.z[inds]
        is_convex_cone(cone, α, s_hat, z_hat, etol)
    end
    return true
end

function is_convex_cone(cone::NonNegativeOrthant, α, s_hat, z_hat, etol=1e-6)
    @assert minimum(s_hat + α * cone.s) >= -etol
    @assert minimum(z_hat + α * cone.z) >= -etol
end

function is_convex_cone(cone::PSDCone, α, s_hat, z_hat, etol=1e-6)
    @assert minimum(eigen(mat((s_hat + α * cone.s))).values) >= -etol
    @assert minimum(eigen(mat((z_hat + α * cone.z))).values) >= -etol
end

function check_affine_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        λ_k = λ[inds]
        circ_λ_k = circ(cone, -λ_k, λ_k)
        @assert isapprox(circ(cone, λ_k, Δzₐ_scaled[inds] + Δsₐ_scaled[inds]), circ_λ_k)
    end
    return true
end

function check_combined_direction(program, λ, Δsₐ_scaled, Δzₐ_scaled, Δs_scaled, Δz_scaled, μ, σ)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        λ_k = λ[inds]
        e = get_e(cone)
        circ_λ_k = circ(cone, -λ_k, λ_k)
        d_s = circ_λ_k - circ(cone, Δsₐ_scaled[inds], Δzₐ_scaled[inds]) + σ * μ * e
        @assert isapprox(circ(cone, λ_k, (Δz_scaled + Δs_scaled)[inds]), d_s)
    end
    return true
end