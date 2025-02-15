#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""
    SecondOrderCone

Represents a second order cone constraint.
"""
mutable struct SecondOrderCone <: Cone
    J::Matrix{Float64}
    W
    inv_W
    p
    v
    w
    s
    z
    λ

    function SecondOrderCone(p)
        cone_socp = new()
        J = Matrix{Float64}(I, p, p)
        J[2:end, 2:end] = -J[2:end, 2:end]
        cone_socp.J = J
        cone_socp.p = p
        return cone_socp
    end
end

function alpha_p(cone::SecondOrderCone)
    α_p = -cone.z[1] - norm(cone.z[2:end])
    return α_p
end

function alpha_d(cone::SecondOrderCone)
    α_d = cone.z[1] - norm(cone.z[2:end])
    return α_d
end

function degree(cone::SecondOrderCone)
    return 1
end

function get_size(cone::SecondOrderCone)
    return cone.p
end

function update(cone::SecondOrderCone, s, z)
    cone.s = s
    cone.z = z
end

function circ(cone::SecondOrderCone, u, v)
    return append!([u' * v], u[1] * v[2:end] + v[1] * u[2:end])
end

function get_step_size(cone::SecondOrderCone, s_scaled, z_scaled)
    J = cone.J
    λ = cone.λ
    p = cone.p
    Q_λ = λ' * J * λ
    c = 1 / sqrt(Q_λ)
    λ_bar = λ * c
    λ_bar_J = λ_bar' * J
    ρ_c = λ_bar_J * s_scaled
    ρ_c_1 = (ρ_c + s_scaled[1]) * 1 / (λ_bar[1] + 1)
    ρ_1 = s_scaled[2:end] - ρ_c_1 .* λ_bar[2:end]
    ρ = zeros(p)
    ρ[1] = ρ_c
    ρ[2:end] = ρ_1
    ρ = c * ρ
    σ_c = λ_bar_J * z_scaled
    σ_c_1 = (σ_c + z_scaled[1]) * 1 / (λ_bar[1] + 1)
    σ_1 = z_scaled[2:end] - σ_c_1 .* λ_bar[2:end]
    σ = zeros(p)
    σ[1] = σ_c
    σ[2:end] = σ_1
    σ = c * σ
    α = inv(maximum((norm(ρ_1) - ρ[1], norm(σ_1) - σ[1])))
    return α
end

function get_scaling_matrix(cone::SecondOrderCone)
    J = cone.J
    s = cone.s
    z = cone.z
    Q_s = s' * J * s
    Q_z = z' * J * z
    s_bar = (1 / sqrt(Q_s)) * s
    z_bar = (1 / sqrt(Q_z)) * z
    γ = sqrt((1 + z_bar' * s_bar) / 2)
    w_bar = 1 / (2 * γ) * (s_bar + J * z_bar)
    v = (1 / sqrt(2 * (w_bar[1] + 1))) * (w_bar + get_e(cone))
    W_bar = 2 * v * v' - J
    Q_w = Q_s / Q_z
    sqrt_Q_w = sqrt(Q_w)
    w = sqrt_Q_w * w_bar
    sqrt_sqrt_Q_w = sqrt(sqrt_Q_w)
    W = sqrt_sqrt_Q_w * W_bar
    λ_c = sqrt(sqrt(Q_s * Q_z))
    bar = s_bar, w_bar, z_bar
    return sqrt_sqrt_Q_w, W, v, w, bar, γ, λ_c
end

function get_inv_scaling_matrix(cone, v, sqrt_sqrt_Q_w)
    J = cone.J
    inv_W_bar = (2 * J * v * v' * J) - J
    inv_W = 1 / sqrt_sqrt_Q_w * inv_W_bar
    return inv_W
end

function get_inv_weighted_mat(cone::SecondOrderCone, V, transpose=false)
    return cone.inv_W * V
end

function get_weighted_mat(cone::SecondOrderCone, V)
    return cone.W * V
end

function get_scaled_lambda(cone::SecondOrderCone, W, bar, γ, λ_c)
    s_bar, w_bar, z_bar = bar
    s_0 = s_bar[1]
    s_1 = s_bar[2:end]
    w_0 = w_bar[1]
    w_1 = w_bar[2:end]
    z_0 = z_bar[1]
    z_1 = z_bar[2:end]
    s_z_1 = s_1 + z_1
    u = w_1' * s_z_1 / (w_0 + 1)
    λ_bar_1 = (1 / 2) * w_1 * ((z_0 - s_0) + u) + s_z_1
    λ_bar = [γ; λ_bar_1]
    λ = λ_c * λ_bar
    return λ
end

function get_scaling_vars(cone::SecondOrderCone)
    sqrt_sqrt_Q_w, W, v, w, bar, γ, λ_c = get_scaling_matrix(cone)
    inv_W = get_inv_scaling_matrix(cone, v, sqrt_sqrt_Q_w)
    cone.v = v
    cone.w = w
    cone.λ = get_scaled_lambda(cone, W, bar, γ, λ_c)
    return W, inv_W, cone.λ
end

function update_scaling_point(cone, J_w, q_bar)
    J = cone.J
    # v is defined on page 9 of coneprog.pdf
    v = cone.v
    W_bar = 2 * v * v' - J
    w = sqrt(J_w) * W_bar * q_bar
    cone.w = w
    w_bar = W_bar * q_bar
    return W_bar, w, w_bar
end

function update_scaling_matrix(cone, J_w, w_bar)
    J = cone.J
    v_new = 1 / sqrt(2 * (w_bar[1] + 1)) * (w_bar + get_e(cone))
    W_bar = 2 * (v_new * v_new') - J
    W = sqrt(J_w) * W_bar
    cone.W = W
    cone.v = v_new
    return J_w
end

function update_inv_scaling_matrix(cone, J_w)
    J = cone.J
    v = cone.v
    inv_W_bar = (2 * J * v * v' * J) - J
    inv_W = 1 / sqrt(J_w) * inv_W_bar
    cone.inv_W = inv_W
end

function update_scaled_lambda(cone, J_s, J_z, W_bar, s_bar, z_bar, w_bar, v, γ)
    J = cone.J
    λ_bar = zeros(cone.p)
    λ_bar[1] = γ
    u = s_bar - J * z_bar
    d = (v[1] * (v' * u) - (u[1] / 2)) / (w_bar[1] + 1)
    c_s_bar = (1 - (d / γ)) / 2
    c_z_bar = (1 + (d / γ)) / 2 * J
    λ_1_bar = W_bar * ((c_s_bar * s_bar) - (c_z_bar * z_bar))
    λ_bar[2:end] = λ_1_bar[2:end]
    λ = sqrt(sqrt(J_s) * sqrt(J_z)) * λ_bar
    cone.λ = λ
end

function update_scaling_vars(cone::SecondOrderCone,
                             s_scaled::AbstractArray{Float64},
                             z_scaled::AbstractArray{Float64},
                             α::Float64)
    # compute common variables
    J = cone.J
    w = cone.w
    s_scaled = cone.λ + α * s_scaled
    z_scaled = cone.λ + α * z_scaled
    J_s = s_scaled' * J * s_scaled
    J_z = z_scaled' * J * z_scaled
    J_q = sqrt(J_s / J_z)
    J_w = w' * J * w * J_q

    # get normalized variables
    s_bar = 1 / sqrt(J_s) * s_scaled
    z_bar = 1 / sqrt(J_z) * z_scaled
    γ = sqrt((1 + z_bar' * s_bar) / 2)
    q_bar = 1 / (2 * γ) * (s_bar + J * z_bar)

    # update scaling vars
    W_bar, w, w_bar = update_scaling_point(cone, J_w, q_bar)
    v = cone.v
    J_w = update_scaling_matrix(cone, J_w, w_bar)
    update_inv_scaling_matrix(cone, J_w)
    update_scaled_lambda(cone, J_s, J_z, W_bar, s_bar, z_bar, w_bar, v, γ)
end

function get_e(cone::SecondOrderCone)
    e = zeros(cone.p)
    e[1] = 1
    return e
end

function get_d_s(cone::SecondOrderCone, s_scaled, z_scaled, b_z, γ, λ, μ, σ)
    function diamond(λ, v)
        λ₀ = λ[1]
        λ₁ = λ[2:end]
        det_A = λ₀^2 - λ₁' * λ₁
        p = length(λ)
        a_4 = (1 / λ₀) * (det_A * I(p-1)) + (λ₁ * λ₁')
        A = [λ₀ -λ₁'; -λ₁ a_4]
        return (1 / det_A) * A * v
    end

    d_s = circ(cone, -λ, λ) - γ * circ(cone, s_scaled, z_scaled) + σ * μ * get_e(cone)
    b_z = b_z - get_weighted_mat(cone, diamond(λ, d_s))
    return b_z
end

export SecondOrderCone