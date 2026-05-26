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

function Base.:in(f::DynamicPolynomials.Polynomial, T::SymmetricGroup)
    return SymmetricGroup(f, T.n)
end