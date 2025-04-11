#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra
using Logging

function lanczos_step(kktsystem, kkt_1_1, inv_M_1, inv_M_2, q_0, q_1, β_0)
    result = kktmatmul(kktsystem, kkt_1_1, (inv_M_2 * q_1))
    s = inv_M_1 * result
    z = s # inv_M * s since M == I
    α_j = dot(s, q_1)
    z = z - (α_j * q_1) - (β_0 * q_0)
    β = sqrt(dot(z, z))
    q = z ./ β
    return α_j, q_0, q_1, q, β
end

function minres(kktsystem, kkt_1_1, b, v, x_0, inv_M_1, inv_M_2, q_0, q_1, eps=1e-3, max_iterations=Inf)
    # Initialize MINRES
    i = 1
    x = x_0
    r_norm = norm(v, 2)
    β_0 = r_norm
    η = β_0
    γ_0 = 1
    γ_1 = γ_0
    σ_0 = 0
    σ_1 = σ_0
    w_0 = zeros(length(x_0))
    w_1 = w_0
    n = minimum((length(x), max_iterations))
    v_0 = q_0
    v_1 = q_1
    @debug "Executing MINRES method"
    while i <= n
        # Lanczos recurrence
        α, v_0, v_1, v, β_1 = lanczos_step(kktsystem, kkt_1_1, inv_M_1, inv_M_2, v_0, v_1, β_0)
        # QR via Givens rotations
        δ = (γ_1 * α) - (γ_0 * σ_1 * β_0)
        ρ_1 = sqrt(δ^2 + β_1^2)
        ρ_2 = σ_1 * α + γ_0 * γ_1 * β_0
        ρ_3 = σ_0 * β_0
        γ = δ / ρ_1
        σ = β_1 / ρ_1
        # Update solution
        w = (v_1 - (ρ_3 * w_0) - (ρ_2 * w_1)) / ρ_1
        x = x + (γ * η * w)
        r_norm = abs(σ) * norm(r_norm, 2)
        η = -σ * η
        # Check convergence
        res = get_residual(kktsystem, kkt_1_1, b, x)
        # @info norm(res, 2)
        # @debug "||r||_2: " r_norm
        if norm(res, 2) <= eps
            return x[:, 1]
        end
        # Update Iterates
        v_0 = v_1
        v_1 = v
        w_0 = w_1
        w_1 = w
        β_0 = β_1
        σ_0 = σ_1
        σ_1 = σ
        γ_0 = γ_1
        γ_1 = γ
        i += 1
    end
    return x
end

function minres_kkt_solve(kktsystem, kkt_1_1, b, x_0, inv_M_1, inv_M_2, eps=0, max_iterations=Inf, preconditioner="ssor")
    q_0 = zeros(length(x_0))
    r_0 = get_residual(kktsystem, kkt_1_1, b, x_0)

    # Lanczos initialization
    z = r_0
    c = sqrt(r_0' * z)
    q_1 = z ./ c

    # Execute MINRES
    x = minres(kktsystem, kkt_1_1, b, r_0, x_0, inv_M_1, inv_M_2, q_0, q_1)

    # Reconstruct solution
    x = inv_M_2 * x
    return x
end

export minres_kkt_solve