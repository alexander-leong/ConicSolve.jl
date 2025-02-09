#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra
using Polynomials

function evaluate_monomials(p, v)
    ps = collect(Polynomials.monomials(p))
    vs = map(kv -> kv[2](v), reduce(vcat, enumerate(ps)))
    return vs
end

export evaluate_monomials