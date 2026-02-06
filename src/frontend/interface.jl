#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.:*
import Base.:+
import Base.:-
import Base.:<=
import Base.:>=
import Base.==
import Base.:∩

using LinearAlgebra

"""
    ConicExpression

Represents an affine or inequality constraint with respect to a cone.

Affine constraint
```math
Ax = b
```
i.e. lhs * x = rhs

Inequality constraint
```math
Gx ≤ₖ h
```
i.e. lhs * x ≤ₖ rhs

``≤ₖ`` is a generalized inequality with respect to cone k
- if ``k`` is the NonNegativeOrthant, this is elementwise nonnegative.
- if ``k`` is the SecondOrderCone, this is the 2-norm less than or equal to some scalar.
- if ``k`` is the PSDCone, this is the variable in terms of the positive semidefinite matrix.
"""
mutable struct ConicExpression{T<:Cone}
    cone::T
    inds::Union{UnitRange{Int64}, Int64}
    lhs::Union{Float64, VecOrMat{Float64}}
    rhs::Union{AbstractArray{Float64}, Float64}

    intersecting_constraints::Vector{ConicExpression}
    link_constraints::Vector{ConicExpression}
    function ConicExpression(cone::T, lhs::Union{Float64, VecOrMat{Float64}}, rhs::Union{AbstractArray{Float64}, Float64}) where T <: Cone
        constraint = new{T}()
        constraint.cone = cone
        constraint.inds = length(size(lhs)) == 2 ? (1:size(lhs, 2)) : 1
        constraint.lhs = lhs
        constraint.rhs = rhs
        constraint.intersecting_constraints = []
        constraint.link_constraints = []
        return constraint
    end
end

export ConicExpression

mutable struct ConeEqualityConstraint{T<:Cone}
    v::ConicExpression{T}

    function ConeEqualityConstraint(v::ConicExpression{T}) where T<:Cone
        constraint = new{T}()
        constraint.v = v
        return constraint
    end
end

mutable struct ConeInequalityConstraint{T<:Cone}
    v::ConicExpression{T}

    function ConeInequalityConstraint(v::ConicExpression{T}) where T<:Cone
        constraint = new{T}()
        constraint.v = v
        return constraint
    end
end

export ConeEqualityConstraint
export ConeInequalityConstraint

function Base.:*(lhs::VecOrMat{Float64}, cone::T) where T <: Cone
    return ConicExpression(cone, lhs, Float64[])
end

function Base.:*(lhs::Float64, cone::T) where T <: Cone
    n = get_size(cone)
    lhs = lhs * ones(n)
    return ConicExpression(cone, lhs, Float64[])
end

function Base.:+(constraint1::ConicExpression{T}, constraint2::ConicExpression{U}) where {T, U <: Cone}
    push!(constraint1.link_constraints, constraint2)
    return constraint1
end

function Base.:-(constraint1::ConicExpression{T}, constraint2::ConicExpression{U}) where {T, U <: Cone}
    constraint2.lhs *= -1
    push!(constraint1.link_constraints, constraint2)
    return constraint1
end

function l1(lhs::VecOrMat{Float64}, cone::Cone)
    p = get_size(cone)
    intersecting_cone = NonNegativeOrthant(p)
    intersecting_constraint = ConicExpression(cone, Matrix{Float64}(I, cone.p, cone.p), zeros(get_size(intersecting_cone)))
    expression = ConicExpression(cone, lhs, Float64[])
    push!(expression.intersecting_constraints, intersecting_constraint)
    return expression
end

function l2(lhs::VecOrMat{Float64}, cone::Cone)
    p = get_size(cone)
    intersecting_cone = SecondOrderCone(p)
    intersecting_constraint = ConicExpression(cone, Matrix{Float64}(I, cone.p, cone.p), zeros(get_size(intersecting_cone)))
    expression = ConicExpression(cone, lhs, Float64[])
    push!(expression.intersecting_constraints, intersecting_constraint)
    return expression
end

function lmi(lhs::Vector{Matrix{Float64}}, cone::PSDCone)
    p = get_size(cone)
    G = zeros((p, length(lhs)))
    for (i, v) in enumerate(lhs)
        G[:, i] = svec(v)
    end
    expression = ConicExpression(cone, G, Float64[])
    return ConeInequalityConstraint(expression)
end

export l1
export l2
export lmi

function Base.:(<=)(cone::NonNegativeOrthant, rhs::Union{AbstractArray{Float64}, Float64})
    return ConeInequalityConstraint{NonNegativeOrthant}(cone, Matrix{Float64}(I, cone.p, cone.p), rhs)
end

function Base.:(<=)(constraint::ConicExpression{SecondOrderCone}, rhs::Float64)
    constraint.rhs = rhs
    return constraint
end

function Base.:(<=)(cone::PSDCone, rhs::Matrix{Float64})
    return ConeInequalityConstraint{PSDCone}(cone, Matrix{Float64}(I, cone.p, cone.p), svec(rhs))
end

function Base.:(<=)(constraint::ConeInequalityConstraint{PSDCone}, rhs::Matrix{Float64})
    constraint.v.rhs = svec(rhs)
    return constraint
end

function Base.:(==)(constraint::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where {T<:Cone}
    constraint.rhs = rhs
    constraint = ConeEqualityConstraint(constraint)
    return constraint
end

# function Base.:∩(cone1::Cone, cone2::Cone)
#     return [cone1, cone2]
# end

# function Base.:∩(cones::Vector{Cone}, cone::Cone)
#     push!(cones, cone)
#     return cones
# end

function Base.getindex(cone::T, inds::Union{UnitRange{Int64}, Int64}) where {T<:Cone}
    constraint = ConicExpression(cone, nothing, nothing)
    constraint.inds = inds
    return constraint
end