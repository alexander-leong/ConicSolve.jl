#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import Base.==
import Base.<=
import Base.>=

mutable struct PolynomialFunction
    basis
    basis_half
    basis_type
    deg
    f
    PolynomialFunction() = new()
    PolynomialFunction(basis,
                        basis_half,
                        basis_type,
                        deg,
                        f) = new(basis,
                            basis_half,
                            basis_type,
                            deg,
                            f)
end

mutable struct PSDExpression
    expression::ConicExpression

    function PSDExpression(lhs::Union{Float64, VecOrMat{Float64}}, rhs::Union{AbstractArray{Float64}, Float64})
        cone = PSDCone(length(rhs))
        expression = ConicExpression(cone, lhs, rhs)
        constraint = new()
        constraint.expression = expression
        return constraint
    end
end

function Base.:(==)(sos_p::SOSPolynomial, rhs::Float64)
    A, b = add_polynomial_equality_constraint(sos_p.sos, rhs, sos_p.p)
    constraint = PSDExpression(A, b)
    constraint.expression.cone = sos_p.sos.cones[end]
    return constraint
end

function Base.:(<=)(sos_p::SOSPolynomial, rhs::Float64)
    sos_p.p *= -1
    A, b, n, _ = get_polynomial_equality_constraint_from_coefficients(sos_p, -rhs)
    constraint = PSDExpression(A, b)
    constraint.expression.cone = sos_p.sos.cones[end]
    return constraint
end

function Base.:(>=)(sos_p::SOSPolynomial, rhs::Float64)
    p = sos_p.p
    A, b, n, _ = get_polynomial_equality_constraint_from_coefficients(sos_p, rhs)
    constraint = PSDExpression(A, b)
    constraint.expression.cone = sos_p.sos.cones[end]
    return constraint
end

function parse_arg(program_int::ProgramInterface, arg::PSDExpression)
    program = program_int.cone_qp
    add_variable(program, arg.expression.cone, arg.expression.cone.p)
    parse_arg(program_int, arg.expression)
    return program_int
end

function parse_obj_arg(program_int::ProgramInterface, arg::DynamicPolynomials.Polynomial)
    program = program_int.cone_qp
    ir = program_int.ir
    vars = program.vars
    for cone in vars.cones
        set_objective(ir, cone, ones(get_size(cone)))
    end
    return program
end

function dispatch(program_int::ProgramInterface, arg::PSDExpression, cones, equalities)
    return dispatch(program_int, arg.expression, cones, equalities)
end

function add_variable(program::ConeQP, ::Type{SOS}, p, variables)
    n = maxdegree(p)
    sos = SOS(n, variables)
    cone = add_variable(program, PSDCone(n), n)
    sos.cones = [cone]
    return sos
end

export add_variable