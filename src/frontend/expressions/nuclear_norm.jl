#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.:*
import Base.==
import Base.:∈
import Base.:in

mutable struct NuclearNorm
    cone::PSDCone

    function NuclearNorm()
        obj = new()
        return obj
    end
    function NuclearNorm(p)
        obj = new()
        obj.cone = PSDCone(p)
        return obj
    end
end

export NuclearNorm

mutable struct NuclearNormExpression
    cone::PSDCone
    constraint::ConicExpression{PSDCone}
    sdp::NuclearNormSDP
    function NuclearNormExpression()
        obj = new()
        return obj
    end
    function NuclearNormExpression(x::Matrix{Float64}, cone::PSDCone)
        obj = new()
        obj.cone = cone
        p = cone.p
        obj.sdp = NuclearNormSDP(x, (p, p))
        return obj
    end
end

function Base.:*(::Type{FixedValue}, constraint::NuclearNorm)
    p = constraint.cone.p
    return NuclearNormExpression(Matrix{Float64}(I, p, p), constraint.cone)
end

function nuclear_norm(x::NuclearNorm)
    return x
end

export nuclear_norm

function Base.:in(x::NuclearNorm, cone::NonNegativeOrthant)
    p = get_size(x.cone)
    cone = NonNegativeOrthant(p)
    lhs, rhs = get_default_inequality_constraint(cone)
    return ConicExpression(cone, lhs, rhs)
end

function Base.:(==)(expression::NuclearNormExpression, rhs::Tuple{AbstractArray{Float64}, T}) where {T<:AbstractArray{Bool}}
    A, b = set_off_diag_constraint(expression.sdp, rhs...)
    p = expression.sdp.num_rows
    expression.cone = PSDCone(p)
    constraint = ConicExpression(expression.cone, A, b)
    expression.constraint = constraint
    return expression
end