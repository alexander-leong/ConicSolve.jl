#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.:*
import Base.:+
import Base.:-
import Base.==
import Base.<=
import Base.>=
import Base.:∈
import Base.:∩

import Base.:in

using LinearAlgebra
using PermutationGroups

include("expressions/conic_expression.jl")
include("expressions/sos_polynomial.jl")
include("expressions/nuclear_norm.jl")
include("expressions/l1_norm.jl")
include("expressions/l2_norm.jl")
include("expressions/symmetric_group.jl")

function Base.:*(lhs::VecOrMat{Float64}, cone::T) where T <: Cone
    return ConicExpression(cone, lhs, Float64[])
end

function Base.:*(lhs::Float64, cone::T) where T <: Cone
    n = get_size(cone)
    lhs = lhs * ones(n)
    return ConicExpression(cone, lhs, Float64[])
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

export minimize

mutable struct TraceNorm
    constraint::ConicExpression{PSDCone}
end

function tr(cone::PSDCone)
end

function lmi(lhs::Vector{Matrix{Float64}}, cone::PSDCone)
    p = get_size(cone)
    G = zeros((p, length(lhs)))
    for (i, v) in enumerate(lhs)
        G[:, i] = svec(v)
    end
    return ConicExpression(cone, G, Float64[])
end

export lmi