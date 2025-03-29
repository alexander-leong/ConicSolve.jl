#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra
using Logging

function lanczos_step(A, inv_M_1, inv_M_2, q_0, q_1, β_0)
    s = inv_M_1 * (A * (inv_M_2 * q_1))
    z = s # inv_M * s where M = I
    α_j = s' * q_1
    z = z - (α_j * q_1) - (β_0 * q_0)
    β = sqrt(s' * z)
    q_0 = q_1
    q_1 = z ./ β
    return α_j, q_0, q_1, z, β
end

function minres(A, b, v, x_0, lanczos_step, inv_M_1, inv_M_2, q_0, q_1, eps=1e-3, max_iterations=Inf)
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
    n = minimum((size(A)[1], max_iterations))
    v_0 = zeros(length(x_0))
    v_1 = b - A * x_0
    @debug "Executing MINRES method"
    while i <= n
        # Lanczos recurrence
        # α, q_0, q_1, v, β_1 = lanczos_step(A, inv_M_1, inv_M_2, q_0, q_1, β_0)
        v_1 = v_1 / β_0
        α = v_1' * A * v_1
        v = (A * v_1) - (α * v_1) - (β_0 * v_0)
        β_1 = norm(v, 2)
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
        @debug "||r||_2: " r_norm
        if r_norm <= eps
            return x
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
    end
    return x
end

function preconditioned_minres(A, b, x_0, eps=1e-3, max_iterations=Inf)
    inv_M_1 = Matrix{Float64}(I, size(A))
    inv_M_2 = Matrix{Float64}(I, size(A))
    q_0 = zeros(length(x_0))
    r_0 = b - A*x_0

    # Lanczos initialization
    z = inv_M_1 * r_0
    c = sqrt(r_0' * z)
    q_1 = z ./ c

    # Execute MINRES
    x = minres(A, b, r_0, x_0, lanczos_step, inv_M_1, inv_M_2, q_0, q_1)

    # Reconstruct solution
    x = inv_M_2 * x
    return x
end

function minres_kkt_solve(kktsystem, b_x, b_y, b_z)
    S, b, x_0 = get_kkt_matrix(kktsystem, b_x, b_y, b_z)
    x = preconditioned_minres(S, b, x_0)
    return x
end

export minres_kkt_solve