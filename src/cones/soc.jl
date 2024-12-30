include("cone.jl")

mutable struct SecondOrderCone <: Cone
    J
    p
    s
    z
    λ

    function SecondOrderCone(p)
        cone_socp = new()
        J = I(p)
        J[2:end, 2:end] = -J[2:end, 2:end]
        cone_socp.J = J
        return cone_socp
    end
end

function degree(cone::SecondOrderCone)
    return 1
end

function get_step_size(cone::SecondOrderCone)
    J = cone.J
    p = cone.p
    s = cone.s
    z = cone.z
    c = 1 / sqrt(λ' * J * λ)
    λ_bar = λ ./ sqrt(λ' * J * λ)
    ρ_c = λ_bar' * J * s
    ρ_0 = c * ρ_c
    ρ_c_1 = (ρ_c + s[1]) * 1 / (λ_bar[1] + 1)
    ρ_1 = s[2:end] - ρ_c_1 .* λ_bar[2:end]
    ρ = zeros(p)
    ρ[1] = ρ_0
    ρ[2:end] = ρ_1
    σ_c = λ_bar' * J * z
    σ_0 = c * σ_c
    σ_c_1 = (σ_c + z[1]) * 1 / (λ_bar[1] + 1)
    σ_1 = z[2:end] - σ_c_1 .* λ_bar[2:end]
    σ = zeros(p)
    σ[1] = σ_0
    σ[2:end] = σ_1
    α = maximum((0, 1 / (norm(ρ_1) - ρ_0), 1 / (norm(σ_1) - σ_0)))
    return α
end

function get_scaling_matrix(cone::SecondOrderCone)
    J = cone.J
    s_bar = (1 / sqrt(s' * J * s)) * s
    z_bar = (1 / sqrt(z' * J * z)) * z
    γ = sqrt((1 + z_bar' * s_bar) / 2)
    w_bar = 1 / (2 * γ) * (s_bar + J * z_bar)
    v = sqrt.(w_bar)
    W_bar = 2 * v * v' - J
    W = sqrt(w_bar' * J * w_bar) * W_bar
    return W
end

function get_inv_scaling_matrix(p, v, w)
    J = cone.J
    inv_W_bar = (2 * J * v * v' * J) - J
    inv_W = 1 / (w' * J * w) * inv_W_bar
    return inv_W
end

function get_scaled_lambda(W, z)
    λ = W * z
    return λ
end

function update_scaling_point(cone::SecondOrderCone, w)
    J = cone.J
    z = 1 / (sqrt(z' * J * z)) .* z
    s = 1 / (sqrt(s' * J * s)) .* s
    γ = sqrt((1 + z's) / 2)
    q_bar = 1 / (2 * γ) * (s + J * z)
    J_s = s' * J * s
    J_z = z' * J * z
    J_w = w' * J * w
    J_q = sqrt(J_s / J_z)
    q = sqrt(J_q) .* q_bar
    z_bar = 1 / sqrt(J_z) .* z
    s_bar = 1 / sqrt(J_s) .* s
    w_bar = 1 / (2 * γ) * (s_bar + J * z_bar)
    # v as defined on page 9 of coneprog.pdf
    v = sqrt.(w_bar)
    w = sqrt(J_w * q) * (2 * v * v' - J) * q_bar
    # pack vars
    Js = (J_q, J_s, J_z)
    bar = s_bar, z_bar
    return Js, bar, v, w
end

function update_scaling_matrix(cone::SecondOrderCone, w, _)
    J = cone.J
    Js, bar, v, w = update_scaling_point(cone, w)
    # unpack vars
    J_q, J_s, J_z = Js
    s_bar, z_bar = bar
    γ = sqrt((1 + z_bar' * s_bar) / 2)
    q_bar = 1 / (2 * γ) * (s_bar + J * z_bar)
    w_bar = (2 * (v * v') - J) * q_bar
    W_bar = 2 * (v * v') - J
    v_new = sqrt.(w_bar)
    J_w = w' * J * w
    W = sqrt(J_w * J_q) * (2 * (v_new * v_new') - J)
    γ = sqrt((1 + z_bar' * s_bar) / 2)
    bar = W_bar, w_bar, bar...
    return J_s, J_z, W, bar, v, w, γ
end

function update_scaled_lambda(cone::SecondOrderCone, w)
    J = cone.J
    result = update_scaling_matrix(cone, w)
    # unpack vars
    J_s, J_z, W, bar, v, w, γ = result
    W_bar, w_bar, s_bar, z_bar = bar
    λ_bar = zeros(p)
    λ_bar[1] = γ
    u = s_bar - J * z_bar
    d = (v[1] * (v' * u) - (u[1] / 2)) / (w_bar[1] + 1)
    c_s_bar = (1 - (d / γ)) / 2
    c_z_bar = (1 + (d / γ)) / 2 * J
    λ_1 = W_bar * ((c_s_bar * s_bar) - (c_z_bar * z_bar))
    λ_bar[2:end] = λ_1[2:end]
    λ = sqrt(sqrt(J_s) * sqrt(J_z)) * λ_bar
    return λ
end

function get_e(cone::SecondOrderCone)
    e = zeros(cone.p)
    e[1] = 1
    return e
end

function get_b_z(cone::SecondOrderCone, s_scaled, z_scaled, W, b_z, γ, μ, σ)
    function circ(u, v)
        return [u' * v, u[1] * v[2:end] + v[1] * u[2:end]]
    end

    function diamond(λ, v)
        λ₀ = λ[1]
        λ₁ = λ[2:end]
        det_A = λ₀^2 - λ₁' * λ₁
        p = length(λ)
        a_4 = (1 / λ₀) * (det_A * I(p-1)) + (λ₁ * λ₁')
        A = [λ₀ -λ₁'; -λ₁ a_4]
        return (1 / det_A) * A * v
    end

    dₛ = circ(-λ, -λ) - γ * circ(s_scaled, z_scaled) + σ * μ * get_e(cone)
    b_z = b_z - W' * diamond(Λ, dₛ)
    return b_z
end