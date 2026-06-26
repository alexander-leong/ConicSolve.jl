#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra

mutable struct L2Norm <: BaseExpression
    λ::Float64
    constraint::ConicExpression{SecondOrderCone}

    intersecting_constraint::Union{IntersectingConstraint{<:Cone}, Nothing}

    function L2Norm(cone::SecondOrderCone,
                    lhs::Matrix{Float64}=Matrix{Float64}(I, get_size(cone), get_size(cone)))
        obj = new()
        obj.constraint = ConicExpression(cone, lhs, zeros(size(lhs, 1)))
        obj.intersecting_constraint = nothing
        return obj
    end
    function L2Norm(cone::SecondOrderCone, lhs::Vector{Float64})
        obj = L2Norm(cone, diagm(lhs))
        return obj
    end
    function L2Norm(cone::NonNegativeOrthant, lhs::VecOrMat{Float64})
        n = get_size(cone)
        aux_var = SecondOrderCone(n)
        obj = L2Norm(aux_var, lhs)
        intersecting_lhs = obj.constraint.lhs
        obj.intersecting_constraint = IntersectingConstraint(aux_var, ConicExpression(cone, intersecting_lhs, zeros(size(lhs, 1))))
        return obj
    end
end

function Base.:*(lhs::Float64, obj::L2Norm)
    obj.λ = lhs
end

function Base.:-(obj::L2Norm)
    obj.λ = -1.
end

function l2(lhs::VecOrMat{Float64}, cone::Cone)
    return L2Norm(cone, lhs)
end

function parse_arg(program_int::ProgramInterface, arg::L2Norm)
    arg.constraint.lhs *= -1
    constraint = isnothing(arg.intersecting_constraint) ? arg.constraint : arg.intersecting_constraint
    parse_arg(program_int, constraint)
end

function parse_obj_arg(program_int::ProgramInterface, arg::L2Norm)
    parse_arg(program_int, arg)
    c = arg.λ * ones(get_size(arg.constraint.cone))
    set_objective(program_int.ir, arg.constraint.cone, c)
    return program_int
end

export l2