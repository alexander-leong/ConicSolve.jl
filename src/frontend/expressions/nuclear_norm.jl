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

mutable struct NuclearNormExpression <: BaseExpression
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

function parse_arg(program_int::ProgramInterface, arg::NuclearNormExpression)
    ir = program_int.ir
    push!(ir._all_affine_constraints, arg.constraint)
    return program_int
end

function parse_obj_arg(program_int::ProgramInterface, arg::NuclearNorm)
    c = get_trace((arg.cone.p, arg.cone.p))
    set_objective(program_int.ir, arg.cone, c)
    return program_int
end

function dispatch(program_int::ProgramInterface, arg::NuclearNormExpression, cones, equalities)
    if !(arg.cone in cones)
        push!(cones, arg.cone)
    end
    return dispatch(program_int, arg.constraint, cones, equalities)
end

function add_variable(program::ConeQP, ::Type{NuclearNorm}, dims)
    obj = NuclearNorm(sum(dims))
    cone = obj.cone
    add_variable(program, cone, cone.p)
    return obj
end

export add_variable