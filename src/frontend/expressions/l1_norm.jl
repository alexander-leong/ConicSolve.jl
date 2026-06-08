#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra

mutable struct L1Norm <: BaseExpression
    λ::Float64
    cone::Cone
    aux_var::Cone
    constraints::Vector{ConicExpression{NonNegativeOrthant}}

    intersecting_constraints::Vector{IntersectingConstraint{<:Cone}}

    function L1Norm(cone::NonNegativeOrthant, lhs::Matrix{Float64})
        obj = new()
        obj.cone = cone
        obj.constraints = ConicExpression{NonNegativeOrthant}[]
        obj.intersecting_constraints = IntersectingConstraint{<:Cone}[]
        p = get_size(cone)
        rhs = zeros(size(lhs, 1))

        aux_var = NonNegativeOrthant(p)
        obj.aux_var = aux_var
        expression_1, expression_3 = get_expressions(cone, aux_var, lhs, rhs)
        push!(obj.constraints, expression_1)
        push!(obj.constraints, expression_3)

        return obj
    end
    function L1Norm(cone::NonNegativeOrthant)
        p = get_size(cone)
        obj = L1Norm(cone, Matrix{Float64}(I, p, p))
        return obj
    end
    function L1Norm(cone::NonNegativeOrthant, lhs::Vector{Float64})
        obj = L1Norm(cone, diagm(lhs))
        return obj
    end
    function L1Norm(cone::SecondOrderCone, lhs::VecOrMat{Float64})
        n = get_size(cone)
        aux_var = NonNegativeOrthant(n)
        obj = new()
        obj.cone = cone
        obj.aux_var = aux_var
        obj.intersecting_constraints = IntersectingConstraint{<:Cone}[]
        constraints = get_expressions(cone, aux_var, lhs)
        for constraint in constraints
            intersecting_lhs = constraint.lhs
            push!(obj.intersecting_constraints, IntersectingConstraint(aux_var, ConicExpression(constraint.cone, intersecting_lhs, zeros(size(constraint.lhs, 1)))))
        end
        return obj
    end
end

function get_expressions(cone::Cone, aux_var::Cone, lhs, rhs=zeros(size(lhs, 1)))
    expression_1 = ConicExpression(cone, lhs, rhs)
    expression_2 = ConicExpression(aux_var, lhs, rhs)
    push!(expression_1.link_constraints, expression_2)
    expression_3 = ConicExpression(cone, -lhs, rhs)
    expression_4 = ConicExpression(aux_var, lhs, rhs)
    push!(expression_3.link_constraints, expression_4)
    return expression_1, expression_3
end

function Base.:*(lhs::Float64, obj::L1Norm)
    obj.λ = lhs
    return obj
end

function Base.:-(obj::L1Norm)
    obj.λ = -1.
end

function l1(lhs::VecOrMat{Float64}, cone::Cone)
    return L1Norm(cone, lhs)
end

function parse_arg(program_int::ProgramInterface, arg::L1Norm)
    # Expression evaluated on whether additional constraint on same cone variable (intersecting constraint)
    # or different cone variable (arg.constraints), not both! in order to handle expressions like
    # e.g. l2(var1) + l1(var1) where var1 ∈ SecondOrderCone × NonNegativeOrthant
    # Don't want to introduce auxiliary variables and constraints to force equality condition!
    constraints = isempty(arg.intersecting_constraints) ? arg.constraints : arg.intersecting_constraints
    for expression in constraints
        constraint = expression.constraint
        constraint.lhs *= -1
        parse_arg(program_int, expression)
    end
end

function parse_obj_arg(program_int::ProgramInterface, arg::L1Norm)
    add_variable(program_int.cone_qp, arg.aux_var, get_size(arg.aux_var))
    parse_arg(program_int, arg)
    c = arg.λ * ones(get_size(arg.aux_var))
    set_objective(program_int, arg.aux_var, c)
    c = zeros(get_size(arg.cone))
    set_objective(program_int, arg.cone, c)
    return program_int
end

export l1