#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

mutable struct L1Norm
    cone::NonNegativeOrthant
    constraints::Vector{ConicExpression{NonNegativeOrthant}}
    lhs::VecOrMat

    function L1Norm(cone::NonNegativeOrthant, lhs::VecOrMat)
        obj = new()
        obj.cone = cone
        obj.lhs = lhs
        p = get_size(cone)
        rhs = zeros(get_size(cone))

        expression = ConicExpression(cone, Matrix{Float64}(I, p, p), rhs)
        intersecting_constraint = IntersectingConstraint(cone, expression)
        push!(constraints, intersecting_constraint)

        intersecting_cone = NonNegativeOrthant(p)
        rhs = zeros(get_size(intersecting_cone))
        expression = ConicExpression(cone, Matrix{Float64}(-I, p, p), rhs)
        intersecting_constraint = IntersectingConstraint(intersecting_cone, expression)
        push!(constraints, intersecting_constraint)

        return obj
    end
end

function l1(lhs::VecOrMat{Float64}, cone::Cone)
    constraint = L1Norm(cone, lhs)
    return constraint
end

function Base.:+(lhs::L1Norm, rhs::L1Norm)
end

function Base.:-(lhs::L1Norm, rhs::L1Norm)
end

export l1