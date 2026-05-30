#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

mutable struct L1Norm
    cone::NonNegativeOrthant
    # implicit constraints
    constraints::Vector{IntersectingConstraint{NonNegativeOrthant}}

    function L1Norm(cone::NonNegativeOrthant, lhs::VecOrMat)
        obj = new()
        obj.cone = cone
        obj.constraints = ConicExpression{NonNegativeOrthant}[]
        p = get_size(cone)
        rhs = zeros(get_size(cone))

        expression = ConicExpression(NonNegativeOrthant(p), lhs, rhs)
        intersecting_constraint = IntersectingConstraint(cone, expression)
        push!(obj.constraints, intersecting_constraint)

        intersecting_cone = NonNegativeOrthant(p)
        rhs = zeros(get_size(intersecting_cone))
        expression = ConicExpression(NonNegativeOrthant(p), -lhs, rhs)
        intersecting_constraint = IntersectingConstraint(intersecting_cone, expression)
        push!(obj.constraints, intersecting_constraint)

        return obj
    end
end

function l1(lhs::VecOrMat{Float64}, cone::Cone)
    constraint = L1Norm(cone, lhs)
    return constraint
end

function Base.:+(lhs::L1Norm, rhs::L1Norm)
    return [lhs, rhs]
end

function Base.:-(lhs::L1Norm, rhs::L1Norm)
    constraint = rhs.constraints[end].constraint
    constraint.lhs *= -1
    return [lhs, rhs]
end

function Base.:+(lhs::Vector{L1Norm}, rhs::L1Norm)
    return [lhs..., rhs]
end

function Base.:-(lhs::Vector{L1Norm}, rhs::L1Norm)
    constraint = rhs.constraints[end].constraint
    constraint.lhs *= -1
    return [lhs..., rhs]
end

function parse_arg(program_int::ProgramInterface, arg::L1Norm)
    expressions = arg.constraints
    for expression in expressions
        constraint = expression.constraint
        add_variable(program_int.cone_qp, constraint.cone, constraint.cone.p)
        parse_arg(program_int, constraint)
    end
end

function parse_arg(program_int::ProgramInterface, arg::Vector{L1Norm})
    for obj in arg
        parse_arg(program_int, obj)
    end
end

function parse_obj_arg(program_int::ProgramInterface, arg::L1Norm)
    set_objective(program_int.ir, arg.cone, ones(get_size(arg.cone)))
    return program_int
end

function parse_obj_arg(program_int::ProgramInterface, arg::Vector{L1Norm})
    last_obj = nothing
    for obj in arg
        last_obj = parse_obj_arg(program_int, obj)
    end
    return last_obj
end

export l1