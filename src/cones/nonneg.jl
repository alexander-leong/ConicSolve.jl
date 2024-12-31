#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("cone.jl")

mutable struct NonNegativeOrthant <: Cone
    W
    inv_W
    p
    s
    z
    λ

    function NonNegativeOrthant(p)
        cone_nonneg = new()
        cone_nonneg.p = p
        return cone_nonneg
    end
end

function alpha_p(cone::NonNegativeOrthant)
    α_p = minimum(-cone.z)
    return α_p
end

function alpha_d(cone::NonNegativeOrthant)
    α_d = minimum(cone.z)
    return α_d
end

function get_mat_size(cone::NonNegativeOrthant)
    return cone.p
end

function get_inv_weighted_mat(cone::NonNegativeOrthant, V, transpose=false)
    return cone.inv_W * V
end

function get_weighted_mat(cone::NonNegativeOrthant, V)
    return cone.W * V
end

function degree(cone::NonNegativeOrthant)
    return cone.p
end

function circ(cone::NonNegativeOrthant, u, v)
    return u .* v
end

function get_step_size(cone::NonNegativeOrthant, s_scaled, z_scaled)
    ρ = circ(cone, inv.(cone.λ), s_scaled)
    σ = circ(cone, inv.(cone.λ), z_scaled)
    min_inv_rho = minimum(ρ)
    min_inv_sigma = minimum(σ)
    α = inv(maximum((-min_inv_rho, -min_inv_sigma)))
    return α
end

function get_scaling_point(cone::NonNegativeOrthant)
    sqrt_s = sqrt.(cone.s)
    inv_sqrt_z = inv.(sqrt.(cone.z))
    w = circ(cone, sqrt_s, inv_sqrt_z)
    return w
end

function update(cone::NonNegativeOrthant, s, z)
    cone.s = s
    cone.z = z
end

function get_scaled_lambda(cone::NonNegativeOrthant)
    sqrt_s = sqrt.(cone.s)
    sqrt_z = sqrt.(cone.z)
    λ = circ(cone, sqrt_z, sqrt_s)
    return λ
end

function update_scaling_point(cone::NonNegativeOrthant, s_scaled, z_scaled, α, λ, w)
    w = circ(cone, circ(cone, sqrt.(λ + α * s_scaled), 1 ./ sqrt.(λ + α * z_scaled)), w)
    return w
end

function update_scaled_lambda(cone::NonNegativeOrthant, s_scaled, z_scaled, α)
    λ = circ(cone, sqrt.(cone.λ + α * z_scaled), sqrt.(cone.λ + α * s_scaled))
    return λ
end

function update_scaling_vars(cone::NonNegativeOrthant,
                             s_scaled::AbstractArray,
                             z_scaled::AbstractArray,
                             α::Float64)
    # from page 24 of coneprog.pdf
    w = update_scaling_point(cone, s_scaled, z_scaled, α, cone.λ, diag(cone.W))
    cone.λ = update_scaled_lambda(cone, s_scaled, z_scaled, α)
    cone.W = Diagonal(w)
    cone.inv_W = Diagonal(inv.(w))
end

function get_scaling_vars(cone::NonNegativeOrthant)
    cone.λ = get_scaled_lambda(cone)
    w = get_scaling_point(cone)
    R = Diagonal(w)
    inv_R = Diagonal(inv.(w))
    return R, inv_R, cone.λ
end

function get_e(cone::NonNegativeOrthant)
    return ones(cone.p)
end

function get_d_s(cone::NonNegativeOrthant, s_scaled, z_scaled, b_z, γ, λ, μ, σ)
    function diamond(Λ, v)
        return inv(Diagonal(Λ)) * v
    end

    dₛ = circ(cone, -λ, λ) - γ * circ(cone, s_scaled, z_scaled) + σ * μ * get_e(cone)
    b_z = b_z - get_weighted_mat(cone, diamond(λ, dₛ))
    return b_z
end

export NonNegativeOrthant