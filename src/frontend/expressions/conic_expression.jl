#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.:+
import Base.:-
import Base.==
import Base.<=
import Base.>=
import Base.:∈
import Base.:in

abstract type BaseExpression end

function Base.:+(lhs::Union{Vector{<:BaseExpression}, T}, rhs::U) where {T<:BaseExpression, U<:BaseExpression}
    return lhs isa Vector ? [lhs..., rhs] : [lhs, rhs]
end

function Base.:-(lhs::Union{Vector{<:BaseExpression}, T}, rhs::U) where {T<:BaseExpression, U<:BaseExpression}
    constraint = -rhs.constraints
    return lhs isa Vector ? [lhs..., constraint] : [lhs, constraint]
end

"""
    ConicExpression

Represents an affine or inequality constraint with respect to a cone.

Affine constraint
```math
Ax = b
```
i.e. lhs * x = rhs

Inequality constraint
```math
Gx ≤ₖ h
```
i.e. lhs * x ≤ₖ rhs

``≤ₖ`` is a generalized inequality with respect to cone k
- if ``k`` is the NonNegativeOrthant, this is elementwise nonnegative.
- if ``k`` is the SecondOrderCone, this is the 2-norm less than or equal to some scalar.
- if ``k`` is the PSDCone, this is the variable in terms of the positive semidefinite matrix.
"""
mutable struct ConicExpression{T<:Cone} <: BaseExpression
    cone::T
    inds::Union{UnitRange{Int64}, Int64}
    lhs::Union{Float64, VecOrMat{Float64}}
    rhs::Union{AbstractArray{Float64}, Float64}

    link_constraints::Vector{ConicExpression}
    function ConicExpression(cone::T, lhs::Union{Float64, VecOrMat{Float64}}, rhs::Union{AbstractArray{Float64}, Float64}) where T <: Cone
        constraint = new{T}()
        constraint.cone = cone
        constraint.inds = length(size(lhs)) == 2 ? (1:size(lhs, 2)) : 1
        constraint.lhs = lhs
        constraint.rhs = rhs
        constraint.link_constraints = []
        return constraint
    end
end

export ConicExpression

"""
    add_to_affine_constraint(constraint, cone, lhs)

Sets the elements of the left hand side, *lhs* of an existing (affine or inequality) constraint with respect to a different cone.
Used to define a constraint involving multiple variables in constraint.cone and cone.

Example:
```math
\\begin{aligned}
x₁ + x₂ = 0 \\\\
x₁ ∈ K₁ \\\\
x₂ ∈ K₂ \\\\
... \\\\
xₙ ∈ Kₙ \\\\
C = K₁ × K₂ × ... Kₙ
\\end{aligned}
```
if *constraint* is a *ConicExpression* object that defines an *lhs* (constraint.lhs), i.e. x₁ in terms of cone K₁,
then *set\\_cone\\_constraint* sets the *lhs*, i.e. x₂

Cone ``C`` is the Cartesian product of all cone variables making up the cone program.
"""
function add_to_affine_constraint(constraint::ConicExpression, cone::Cone, lhs::VecOrMat{Float64})
    link_constraint = ConicExpression(cone, lhs, zeros(size(lhs, 1)))
    push!(constraint.link_constraints, link_constraint)
end

mutable struct ConicInequalityExpression{T<:Cone}
    expression::ConicExpression{T}
end

function Base.:+(constraint1::ConicExpression{T}, constraint2::ConicExpression{U}) where {T, U <: Cone}
    push!(constraint1.link_constraints, constraint2)
    return constraint1
end

function Base.:-(constraint1::ConicExpression{T}, constraint2::ConicExpression{U}) where {T, U <: Cone}
    constraint2.lhs *= -1
    push!(constraint1.link_constraints, constraint2)
    return constraint1
end

function Base.:+(constraint::ConicExpression{T}, rhs::Vector{Float64}) where {T<:Cone}
    constraint.rhs = rhs
    return constraint
end

function Base.:-(constraint::ConicExpression{T}, rhs::Vector{Float64}) where {T<:Cone}
    constraint.rhs = -rhs
    return constraint
end

function Base.:(<=)(expression::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where T<:Cone
    expression.lhs *= -1
    expression.rhs = -rhs
    expression = ConicInequalityExpression(expression)
    return expression
end

function Base.:(>=)(expression::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where T<:Cone
    expression.rhs = rhs
    expression = ConicInequalityExpression(expression)
    return expression
end

function Base.:(==)(expression::ConicExpression{T}, rhs::Union{AbstractArray{Float64}, Float64}) where T<:Cone
    expression.rhs = rhs
    if typeof(rhs) <: Vector{Float64}
        return expression
    end
    if length(rhs) == 1
        if size(expression.lhs, 1) > 1
            expression.rhs = rhs * ones(size(expression.lhs, 1))
        else
            expression.rhs = [rhs]
        end
    end
    return expression
end

function Base.:in(expression::ConicExpression{T}, cone::NonNegativeOrthant) where T<:Cone
    expression.rhs = zeros(get_size(cone))
    intersecting_constraint = IntersectingConstraint(cone, expression)
    return intersecting_constraint
end

function Base.:in(expression::ConicExpression{T}, cone::SecondOrderCone) where T<:Cone
    expression.rhs = zeros(get_size(cone))
    intersecting_constraint = IntersectingConstraint(cone, expression)
    return intersecting_constraint
end

function Base.:in(expression::ConicExpression{T}, cone::PSDCone) where T<:Cone
    expression.rhs = zeros(get_size(cone))
    intersecting_constraint = IntersectingConstraint(cone, expression)
    return intersecting_constraint
end

function Base.:in(constraint::ConicExpression{PSDCone}, V::Matrix{Float64})
    constraint.rhs = svec(V)
    return constraint
end

function minimize(args::Vector{ConicExpression{T}}) where T<:Cone
    obj = ObjectiveFunction(args...)
    return obj
end

mutable struct IntersectingConstraint{T<:Cone, U<:Cone}
    cone::T
    constraint::ConicExpression{U}
    function IntersectingConstraint(cone::T, constraint::ConicExpression{U}) where {T<:Cone, U<:Cone}
        obj = new{T, U}()
        obj.cone = cone
        obj.constraint = constraint
        return obj
    end
end

function Base.:(==)(expression::IntersectingConstraint{T, U}, rhs::Union{AbstractArray{Float64}, Float64}) where {T<:Cone, U<:Cone}
    constraint = expression.constraint
    constraint.rhs = rhs
    return expression
end

const AffineConstraints = Vector{ConicExpression{<:Cone}}
const InequalityConstraints = Vector{Union{ConicExpression{<:Cone}, IntersectingConstraint}}
const AllConstraints = Vector{Union{ConicExpression{<:Cone}, IntersectingConstraint}}

export AllConstraints

mutable struct ConeQP_IR
    obj::Vector{PrimalObjective}
    _all_affine_constraints::AllConstraints
    _all_inequality_constraints::AllConstraints

    ids_cones::Vector{UInt64}
    ids_implicit_cones::Vector{UInt64}
    num_slack_vars::UInt64
    num_vars::UInt64
    function ConeQP_IR()
        ir = new()
        ir.obj = []
        ir._all_affine_constraints = AllConstraints()
        ir._all_inequality_constraints = AllConstraints()
        ir.ids_implicit_cones = []
        ir.num_slack_vars = 0
        ir.num_vars = 0
        return ir
    end
end

mutable struct ProgramInterface
    cone_qp::ConeQP
    ir::ConeQP_IR
    function ProgramInterface(cone_qp::ConeQP)
        obj = new()
        obj.cone_qp = cone_qp
        obj.ir = ConeQP_IR()
        return obj
    end
    function ProgramInterface(cone_qp::ConeQP, ir::ConeQP_IR)
        obj = new()
        obj.cone_qp = cone_qp
        obj.ir = ir
        return obj
    end
end

export ProgramInterface

function parse_arg(program_int::ProgramInterface, arg::Vector{<:BaseExpression})
    for obj in arg
        parse_arg(program_int, obj)
    end
end

function parse_obj_arg(program_int::ProgramInterface, arg::Vector{<:BaseExpression})
    last_obj = nothing
    for obj in arg
        last_obj = parse_obj_arg(program_int, obj)
    end
    return last_obj
end

function parse_arg(program_int::ProgramInterface, arg::ConicExpression{<:Cone})
    push!(program_int.ir._all_affine_constraints, arg)
    return program_int
end

function parse_arg(program_int::ProgramInterface, arg::IntersectingConstraint{<:Cone, <:Cone})
    push!(program_int.ir._all_inequality_constraints, arg.constraint)
    return program_int
end

function parse_obj_arg(program_int::ProgramInterface, arg::ConicExpression{<:Cone})
    set_objective(program_int.ir, arg.cone, arg.lhs)
    return program_int
end

function parse_obj_arg(program_int::ProgramInterface, arg::Tuple{Vector{Float64}, IntersectingConstraint{<:Cone, <:Cone}})
    c, intersecting_constraint = arg
    set_objective(program_int.ir, intersecting_constraint.constraint.cone, c)
    program = program_int.cone_qp
    aux_vars = program.aux_vars
    push!(aux_vars.cones, intersecting_constraint.cone)
    parse_arg(program_int.cone_qp, intersecting_constraint)
    return program_int
end

function dispatch(program_int::ProgramInterface, arg::ConicExpression, cones, equalities)
    if !(arg.cone in cones)
        push!(cones, arg.cone)
    end
    return parse_arg(program_int, arg)
end

function dispatch(program_int::ProgramInterface, arg::IntersectingConstraint{<:Cone, <:Cone}, cones, equalities)
    return parse_arg(program_int, arg)
end

function parse_arg(program_int::ProgramInterface, arg::ConicInequalityExpression{NonNegativeOrthant})
    expression = arg.expression
    cone = expression.cone
    # express inequality constraint as equality constraint using slack variables
    add_slack_variable(program_int.cone_qp, cone, cone.p)
    push!(program_int.ir._all_affine_constraints, arg.expression)
    return program_int
end

function dispatch(program_int::ProgramInterface, arg::ConicInequalityExpression, cones, equalities)
    if !(arg.expression.cone in cones)
        push!(cones, arg.expression.cone)
    end
    # process equalities at the end so added slack variables can be separated out later
    push!(equalities, arg)
    return program_int
end
