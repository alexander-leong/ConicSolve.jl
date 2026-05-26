#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.:+
import Base.:-
import Base.==
import Base.<=
import Base.>=
import Base.:∈
import Base.:in

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
    function ConicExpression(cone::T, lhs::Union{Float64, VecOrMat{Float64}}, rhs::Union{AbstractArray{Float64}, Float64}) where T <: Cone
        constraint = new{T}()
        constraint.cone = cone
        constraint.inds = length(size(lhs)) == 2 ? (1:size(lhs, 2)) : 1
        constraint.lhs = lhs
        constraint.rhs = rhs
        constraint.link_constraints = []
        return constraint
    end
end

export ConicExpression

mutable struct ConicInequalityExpression{T<:Cone}
    expression::ConicExpression{T}
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

function Base.:(<=)(expression::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where T<:Cone
    expression.lhs *= -1
    expression.rhs = -rhs
    expression = ConicInequalityExpression(expression)
    return expression
end

function Base.:(>=)(expression::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where T<:Cone
    expression.rhs = rhs
    expression = ConicInequalityExpression(expression)
    return expression
end

function Base.:(==)(expression::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where T<:Cone
    expression.rhs = rhs
    if typeof(rhs) <: Vector{Float64}
        return expression
    end
    if length(rhs) == 1
        if size(expression.lhs, 1) > 1
            expression.rhs = rhs * ones(size(expression.lhs, 1))
        else
            expression.rhs = [rhs]
        end
    end
    return expression
end

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

function minimize(args::Vector{ConicExpression{T}}) where T<:Cone
    obj = ObjectiveFunction(args...)
    return obj
end

mutable struct IntersectingConstraint{T<:Cone, U<:Cone}
    cone::T
    constraint::ConicExpression{U}
    function IntersectingConstraint(cone::T, constraint::ConicExpression{U}) where {T<:Cone, U<:Cone}
        obj = new{T, U}()
        obj.cone = cone
        obj.constraint = constraint
        return obj
    end
end

function Base.:(==)(expression::IntersectingConstraint{T, U}, rhs::Union{AbstractArray{Float64}, Float64}) where {T<:Cone, U<:Cone}
    constraint = expression.constraint
    constraint.rhs = rhs
    return expression
end
