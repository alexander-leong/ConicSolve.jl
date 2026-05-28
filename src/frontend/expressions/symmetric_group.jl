#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.:in

using DynamicPolynomials

mutable struct SymmetricGroup
    f::DynamicPolynomials.Polynomial
    n::Int
    function SymmetricGroup(f, n)
        obj = new()
        obj.f = f
        obj.n = n
        return obj
    end
    function SymmetricGroup(n)
        obj = new()
        obj.n = n
        return obj
    end
end

mutable struct SymmetricGroupAction <: ConicGroupAction
    f::PolynomialFunction
    g::SymmetricGroup
    pg::PermGroup
end

export SymmetricGroupAction

function Base.:in(f::DynamicPolynomials.Polynomial, T::SymmetricGroup)
    return SymmetricGroup(f, T.n)
end

function parse_arg(program_int::ProgramInterface, arg::SymmetricGroup)
    return wedderburn_decompose!(program_int, arg)
end

function parse_obj_arg(program::SymmetryReducedConeQP{SymmetricGroupAction}, arg::DynamicPolynomials.Polynomial)
    program_int = program.program_int
    parse_obj_arg(program_int, arg)
    return program
end

function dispatch(program_int::ProgramInterface, arg::SymmetricGroup, cones, equalities)
    return parse_arg(program_int, arg)
end