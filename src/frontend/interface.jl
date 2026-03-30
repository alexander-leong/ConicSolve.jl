#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.:*
import Base.:+
import Base.:-
import Base.==
import Base.:∈
import Base.:∩

import Base.:in

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

    link_constraints::Vector{ConicExpression}
    # _remap_constraint::Bool
    function ConicExpression(cone::T, lhs::Union{Float64, VecOrMat{Float64}}, rhs::Union{AbstractArray{Float64}, Float64}) where T <: Cone
        constraint = new{T}()
        constraint.cone = cone
        constraint.inds = length(size(lhs)) == 2 ? (1:size(lhs, 2)) : 1
        constraint.lhs = lhs
        constraint.rhs = rhs
        constraint.link_constraints = []
        # constraint._remap_constraint = false
        return constraint
    end
end

export ConicExpression

mutable struct IntersectingConstraint{T<:Cone, U<:Cone}
    cone::T
    constraint::ConicExpression{U}
    # _remap_constraint::Bool
    function IntersectingConstraint(cone::T, constraint::ConicExpression{U}) where {T<:Cone, U<:Cone}
        obj = new{T, U}()
        obj.cone = cone
        obj.constraint = constraint
        # obj._remap_constraint = false
        return obj
    end
end

mutable struct NuclearNormConstraint
    constraint::ConicExpression{PSDCone}
    # _remap_constraint::Bool
    function NuclearNormConstraint(constraint::ConicExpression{PSDCone})
        obj = new()
        obj.constraint = constraint
        # obj._remap_constraint = true
    end
end

mutable struct TraceNorm
    constraint::ConicExpression{PSDCone}
end

mutable struct ObjectiveFunction
    args::Vector

    function ObjectiveFunction(args)
        obj = new()
        obj.args = args
        return obj
    end
end

export ObjectiveFunction

function minimize(args...)
    obj = ObjectiveFunction([args...])
    return obj
end

function minimize(args::Vector{ConicExpression{T}}) where T<:Cone
    obj = ObjectiveFunction(args...)
    return obj
end

export minimize

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

function Base.:+(constraint::ConicExpression{T}, rhs::Vector{Float64}) where {T<:Cone}
    constraint.rhs = rhs
    return constraint
end

function Base.:-(constraint::ConicExpression{T}, rhs::Vector{Float64}) where {T<:Cone}
    constraint.rhs = -rhs
    return constraint
end

function l1(lhs::Vector{Float64}, cone::Cone)
    p = get_size(cone)
    intersecting_cone = NonNegativeOrthant(p)
    constraint_lhs = Matrix{Float64}(I, p, p)
    intersecting_constraint = ConicExpression(cone, constraint_lhs, zeros(get_size(intersecting_cone)))
    intersecting_constraint.rhs = zeros(p)
    expression = IntersectingConstraint(intersecting_cone, intersecting_constraint)
    return lhs, expression
end

function l2(lhs::Vector{Float64}, cone::Cone)
    p = get_size(cone)
    intersecting_cone = SecondOrderCone(p)
    constraint_lhs = Matrix{Float64}(I, p, p)
    intersecting_constraint = ConicExpression(cone, constraint_lhs, zeros(get_size(intersecting_cone)))
    intersecting_constraint.rhs = zeros(p)
    expression = IntersectingConstraint(intersecting_cone, intersecting_constraint)
    return lhs, expression
end

function lmi(lhs::Vector{Matrix{Float64}}, cone::PSDCone)
    p = get_size(cone)
    G = zeros((p, length(lhs)))
    for (i, v) in enumerate(lhs)
        G[:, i] = svec(v)
    end
    return ConicExpression(cone, G, Float64[])
end

function nuclear_norm(cone::PSDCone)
    expression = ConicExpression(cone, Float64[], Float64[])
    expression._remap_constraint = true
    return expression
end

function tr(cone::PSDCone)
end

export l1
export l2
export lmi

function Base.:in(expression::ConicExpression{T}, cone::NonNegativeOrthant) where T<:Cone
    expression.rhs = zeros(get_size(cone))
    intersecting_constraint = IntersectingConstraint(cone, expression)
    return intersecting_constraint
end

function Base.:in(expression::ConicExpression{T}, cone::SecondOrderCone) where T<:Cone
    expression.rhs = zeros(get_size(cone))
    intersecting_constraint = IntersectingConstraint(cone, expression)
    return intersecting_constraint
end

function Base.:in(expression::ConicExpression{T}, cone::PSDCone) where T<:Cone
    expression.rhs = zeros(get_size(cone))
    intersecting_constraint = IntersectingConstraint(cone, expression)
    return intersecting_constraint
end

function Base.:in(constraint::ConicExpression{PSDCone}, V::Matrix{Float64})
    constraint.rhs = svec(V)
    return constraint
end

function Base.:(==)(expression::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where T<:Cone
    expression.rhs = rhs
    return expression
end

function Base.:(==)(expression::IntersectingConstraint{T, U}, rhs::Union{AbstractArray{Float64}, Float64}) where {T<:Cone, U<:Cone}
    constraint = expression.constraint
    constraint.rhs = rhs
    return expression
end

function Base.:(==)(cone::T, rhs::Vector{Float64}) where T<:Cone
    n = get_size(cone)
    constraint = ConicExpression(cone, Matrix{Float64}(I, n, n), rhs)
    return constraint
end

function Base.getindex(cone::T, inds::Union{UnitRange{Int64}, Int64}) where {T<:Cone}
    constraint = ConicExpression(cone, nothing, nothing)
    constraint.inds = inds
    return constraint
end