#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

mutable struct L2Norm
    cone::SecondOrderCone
    constraint::ConicExpression{SecondOrderCone}
    lhs::VecOrMat

    function L2Norm(cone::Cone, lhs::VecOrMat)
        obj = new()
        obj.cone = cone
        obj.lhs = lhs
        p = get_size(cone)
        intersecting_cone = SecondOrderCone(p)
        constraint_lhs = Matrix{Float64}(I, p, p)
        intersecting_constraint = ConicExpression(cone, constraint_lhs, zeros(get_size(intersecting_cone)))
        intersecting_constraint.rhs = zeros(p)
        obj.constraint = IntersectingConstraint(intersecting_cone, intersecting_constraint)
        return obj
    end
end

function l2(lhs::VecOrMat{Float64}, cone::Cone)
    return L2Norm(cone, lhs)
end

function Base.:+(lhs::L2Norm, rhs::L2Norm)
end

function Base.:-(lhs::L2Norm, rhs::L2Norm)
end

export l2