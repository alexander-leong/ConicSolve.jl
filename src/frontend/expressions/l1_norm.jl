#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

mutable struct L1Norm
    cone::NonNegativeOrthant
    constraints::Vector{ConicExpression{NonNegativeOrthant}}

    function L1Norm(cone::NonNegativeOrthant)
        obj = new()
        obj.cone = cone
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

function l1(lhs::Vector{Float64}, cone::Cone)
    constraint = L1Norm(cone)
    return lhs, constraint
end

export l1