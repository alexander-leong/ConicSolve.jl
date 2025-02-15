#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra

"""
    PSDCone

Represents a positive semidefinite cone constraint.
"""
mutable struct PSDCone <: Cone
    W
    inv_W
    p
    s
    z
    λ
    _svd

    function PSDCone(p)
        cone_psd = new()
        cone_psd.p = p
        function svdfact(A)
            return svd(A, alg=LinearAlgebra.QRIteration())
        end
        cone_psd._svd = svdfact
        return cone_psd
    end
end

function alpha_p(cone::PSDCone)
    α_p = minimum(eigen(mat(-cone.z)).values)
    return α_p
end

function alpha_d(cone::PSDCone)
    α_d = minimum(eigen(mat(cone.z)).values)
    return α_d
end

function get_size(cone::PSDCone)
    return Int((cone.p * (cone.p + 1)) / 2)
end

function get_inv_weighted_mat(cone::PSDCone,
                              V::AbstractArray,
                              transpose=false)
    inv_R = cone.inv_W
    ncols = length(size(V)) == 1 ? 1 : size(V)[2]
    W = zeros((size(V)[1], ncols))
    for j in 1:ncols
        if transpose == true
            W[:, j] = svec(inv_R' * mat(V[:, j]) * inv_R)
        else
            W[:, j] = svec(inv_R * mat(V[:, j]) * inv_R')
        end
    end
    return W
end

function get_weighted_mat(cone::PSDCone,
                          V::AbstractArray,
                          transpose=false)
    R = cone.W
    ncols = length(size(V)) == 1 ? 1 : size(V)[2]
    W = zeros((size(V)[1], ncols))
    for j in 1:ncols
        if transpose == true
            W[:, j] = svec(R * mat(V[:, j]) * R')
        else
            W[:, j] = svec(R' * mat(V[:, j]) * R)
        end
    end
    return W
end

function degree(cone::PSDCone)
    return cone.p
end

function update(cone::PSDCone, s, z)
    cone.s = s
    cone.z = z
end

function get_scaling_factors(cone::PSDCone)
    L₁_L = cholesky(mat(cone.s)).L
    L₂_U = cholesky(mat(cone.z)).U
    U, S, V = cone._svd(L₂_U * L₁_L)
    return L₁_L, L₂_U, U, S, V'
end

function update_scaling_vars(cone::PSDCone,
                             s_scaled::AbstractArray{Float64},
                             z_scaled::AbstractArray{Float64},
                             α::Float64)
    # from page 26 of coneprog.pdf
    mat_s_scaled = mat(s_scaled)
    mat_z_scaled = mat(z_scaled)
    Λ = mat(cone.λ)
    L₁_L = cholesky(Λ + α * mat_s_scaled).L
    L₂_U = cholesky(Λ + α * mat_z_scaled).U
    U, S, V = cone._svd(L₂_U * L₁_L)
    cone.λ = svec(Diagonal(S))
    sqrt_λ = sqrt.(S)
    inv_sqrt_λ = inv.(sqrt_λ)
    inv_sqrt_Λ = Diagonal(inv_sqrt_λ)
    R = cone.W * L₁_L * V * inv_sqrt_Λ
    inv_R = inv_sqrt_Λ * U' * L₂_U * cone.inv_W
    cone.W = R
    cone.inv_W = inv_R
end

function get_scaled_value(R, v)
    return svec(R' * mat(v) * R)
end

function get_scaling_vars(cone::PSDCone)
    L₁_L, L₂_U, U, λ, Vt = get_scaling_factors(cone)
    cone.λ = svec(Diagonal(λ))
    sqrt_λ = sqrt.(λ)
    inv_sqrt_λ = inv.(sqrt_λ)
    # from page 10 of coneprog.pdf
    Rz = L₁_L * Vt' * Diagonal(inv_sqrt_λ)
    inv_Rs = Diagonal(inv_sqrt_λ) * U' * L₂_U
    return Rz, inv_Rs, cone.λ
end

function get_step_size(cone::PSDCone, s_scaled, z_scaled)
    λ = cone.λ
    inv_λ = inv.(diag(mat(λ)))
    inv_sqrt_λ = sqrt.(inv_λ)
    inv_sqrt_Λ = Diagonal(inv_sqrt_λ)
    mat_s_scaled = mat(s_scaled)
    ρ = svec(inv_sqrt_Λ * mat_s_scaled * inv_sqrt_Λ)
    mat_z_scaled = mat(z_scaled)
    σ = svec(inv_sqrt_Λ * mat_z_scaled * inv_sqrt_Λ)
    γ_s = eigen(mat(ρ)).values
    γ_z = eigen(mat(σ)).values
    α = inv(maximum((-minimum(γ_s), -minimum(γ_z))))
    return α
end

function get_e(cone::PSDCone)
    p = cone.p
    return svec(I(p))
end

function circ(cone::PSDCone, u, v)
    return (1/2)*svec(mat(u)*mat(v) + mat(v)*mat(u))
end

function get_d_s(cone::PSDCone, s_scaled, z_scaled, b_z, γ, λ, μ, σ)
    function diamond(Λ, v)
        # following eq. 22 of coneprog.pdf
        mat_Λ = mat(Λ)
        diag_mat_Λ = diag(mat_Λ)
        Γ = 2 ./ (diag_mat_Λ .+ diag_mat_Λ')
        return svec(mat(v) .* Γ)
    end

    dₛ = circ(cone, -λ, λ) - γ * circ(cone, s_scaled, z_scaled) + σ * μ * get_e(cone)
    b_z = b_z - get_weighted_mat(cone, diamond(λ, dₛ), true)
    return b_z
end

export PSDCone