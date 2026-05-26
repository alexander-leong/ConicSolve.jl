#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.==
import Base.<=
import Base.>=

mutable struct PSDExpression
    expression::ConicExpression

    function PSDExpression(lhs::Union{Float64, VecOrMat{Float64}}, rhs::Union{AbstractArray{Float64}, Float64})
        cone = PSDCone(length(rhs))
        expression = ConicExpression(cone, lhs, rhs)
        constraint = new()
        constraint.expression = expression
        return constraint
    end
end

function Base.:(==)(sos_p::SOSPolynomial, rhs::Float64)
    A, b = add_polynomial_equality_constraint(sos_p.sos, rhs, sos_p.p)
    constraint = PSDExpression(A, b)
    constraint.expression.cone = sos_p.sos.cones[end]
    return constraint
end

function Base.:(<=)(sos_p::SOSPolynomial, rhs::Float64)
    sos_p.p *= -1
    A, b, n, _ = get_polynomial_equality_constraint_from_coefficients(sos_p, -rhs)
    constraint = PSDExpression(A, b)
    constraint.expression.cone = sos_p.sos.cones[end]
    return constraint
end

function Base.:(>=)(sos_p::SOSPolynomial, rhs::Float64)
    p = sos_p.p
    A, b, n, _ = get_polynomial_equality_constraint_from_coefficients(sos_p, rhs)
    constraint = PSDExpression(A, b)
    constraint.expression.cone = sos_p.sos.cones[end]
    return constraint
end