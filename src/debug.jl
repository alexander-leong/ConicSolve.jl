include("./cones/nonneg.jl")
include("./cones/psdcone.jl")

using LinearAlgebra

function is_convex_cone(program, α)
    for (k, cone) in enumerate(program.cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        s_hat = program.s[inds]
        z_hat = program.KKT_x[program.inds_h][inds]
        is_convex_cone(cone, α, s_hat, z_hat)
    end
    return true
end

function is_convex_cone(cone::NonNegativeOrthant, α, s_hat, z_hat)
    @assert minimum(s_hat + α * cone.s) >= 0
    @assert minimum(z_hat + α * cone.z) >= 0
end

function is_convex_cone(cone::PSDCone, α, s_hat, z_hat)
    @assert minimum(eigen(mat((s_hat + α * cone.s))).values) >= 0
    @assert minimum(eigen(mat((z_hat + α * cone.z))).values) >= 0
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