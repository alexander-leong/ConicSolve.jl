#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

function parse_arg(program::ConeQP, arg::Cone)
    add_variable(program, arg, arg.p)
    return program
end

function parse_arg(program::ConeQP, arg::ConicExpression{<:Cone})
    ir = program.program_ir
    push!(ir._all_affine_constraints, arg)
    return program
end

function parse_arg(program::ConeQP, arg::PSDExpression)
    add_variable(program, arg.expression.cone, arg.expression.cone.p)
    parse_arg(program, arg.expression)
    return program
end

function parse_arg(program::ConeQP, arg::IntersectingConstraint{<:Cone, <:Cone})
    aux_vars = program.aux_vars
    push!(aux_vars.cones, arg.cone)
    ir = program.program_ir
    push!(ir._all_inequality_constraints, arg.constraint)
    return program
end

function remap_constraint(program::ConeQP, arg::ConicExpression{PSDCone})
end

function parse_arg(program::ConeQP, arg::NuclearNormConstraint)
    remap_constraint(program, arg.constraint)
    return program
end

function parse_arg(program::ConeQP, arg::SymmetricGroup)
    program = wedderburn_decompose!(program, arg)
    return program
end

function parse_obj_arg(program::ConeQP, arg::ConicExpression{<:Cone})
    set_objective(program, arg.lhs, arg.cone)
    return program
end

function parse_obj_arg(program::ConeQP, arg::DynamicPolynomials.Polynomial)
    vars = program.vars
    for cone in vars.cones
        set_objective(program, cone, ones(get_size(cone)))
    end
    return program
end

function parse_obj_arg(program::SymmetryReducedConeQP{SymmetricGroupAction}, arg::DynamicPolynomials.Polynomial)
    cone_qp = parse_obj_arg(program.cone_qp, arg)
    program.cone_qp = cone_qp
    return program
end

function parse_obj_arg(program::ConeQP, arg::Tuple{Vector{Float64}, IntersectingConstraint{<:Cone, <:Cone}})
    c, intersecting_constraint = arg
    set_objective(program, intersecting_constraint.constraint.cone, c)
    parse_arg(program, intersecting_constraint)
    return program
end

function define_program(program::ConeQP, obj::ObjectiveFunction, args...)
    out_program = program
    for arg in args
        out_program = parse_arg(program, arg)
    end
    for obj_term in obj.args
        out_program = parse_obj_arg(out_program, obj_term)
    end
    vars = program.vars
    for cone in vars.cones
        add_default_inequality_constraint(program, cone)
    end
    return out_program
end

function define_program(program::ConeQP, args...)
    vars = program.vars
    for cone in vars.cones
        set_objective(program, cone, zeros(Float64, get_size(cone)))
    end
    obj = ObjectiveFunction([])
    return define_program(program, obj, args...)
end

export define_program

function build_program(program::ConeQP, allocated=false)
    set_cones_inds(program)
    
    vars = program.vars
    ir = program.program_ir
    ir.ids_cones = [objectid(cone) for cone in vars.cones]
    P, c = get_primal_objective(program)

    A, b = get_affine_constraint_matrix(program)
    G, h = get_inequality_constraint_matrix(program, allocated)
    program.A = A
    program.G = G
    program.P = P
    program.b = b
    program.c = c
    program.h = h
    return program
end

function build_program(program::SymmetryReducedConeQP, allocated=false)
    build_program(program.cone_qp, allocated)
    return program
end

export build_program