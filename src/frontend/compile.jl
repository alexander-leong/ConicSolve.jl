#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("interface.jl")
include("../program.jl")

function parse_arg(program::ConeQP, arg::Cone)
    add_variable(program, arg, arg.p)
end

function parse_arg(program::ConeQP, arg::ConicExpression{<:Cone})
    ir = program.program_ir
    push!(ir._all_affine_constraints, arg)
end

function parse_arg(program::ConeQP, arg::IntersectingConstraint{<:Cone, <:Cone})
    aux_vars = program.aux_vars
    push!(aux_vars.cones, arg.cone)
    ir = program.program_ir
    push!(ir._all_inequality_constraints, arg.constraint)
end

function parse_arg(program::ConeQP, arg::NuclearNormConstraint)
    remap_constraint(program, arg.constraint)
end

function parse_obj_arg(program::ConeQP, arg::ConicExpression{<:Cone})
    set_objective(program, arg.lhs, arg.cone)
    return arg
end

function parse_obj_arg(program::ConeQP, arg::Tuple{Vector{Float64}, IntersectingConstraint{<:Cone, <:Cone}})
    c, intersecting_constraint = arg
    set_objective(program, intersecting_constraint.constraint.cone, c)
    parse_arg(program, intersecting_constraint)
    return intersecting_constraint
end

function remap_constraint(program::ConeQP, arg::ConicExpression{PSDCone})
end

function define_program(program::ConeQP, obj::ObjectiveFunction, args...)
    # remapping_constraints = []
    for obj_term in obj.args
        parse_obj_arg(program, obj_term)
        # if expression._remap_constraint == true
            # push!(remapping_constraints, expression)
        # end
    end
    vars = program.vars
    for cone in vars.cones
        add_default_inequality_constraint(program, cone)
    end
    for (i, arg) in enumerate(args)
        parse_arg(program, arg)
        # FIXME: throw exception if invalid expression
        # else
        #     throw(ArgumentError("Invalid argument type $(typeof(arg)) at position $(i), must be either a [Cone, ConicExpression] type"))
        # end
    end
end

function define_program(program::ConeQP, args...)
    vars = program.vars
    for cone in vars.cones
        set_objective(program, cone, zeros(Float64, get_size(cone)))
    end
    obj = ObjectiveFunction([])
    define_program(program, obj, args...)
end

export define_program

function build_program(program::ConeQP, allocated=false)
    set_cones_inds(program)
    
    vars = program.vars
    ir = program.program_ir
    ir.ids_cones = [objectid(cone) for cone in vars.cones]
    P, c = get_primal_objective(program)

    A, b = get_affine_constraint_matrix(program)
    # println("Condition number of equality constraint matrix: $(cond(A))")
    G, h = get_inequality_constraint_matrix(program, allocated)
    program.A = A
    program.G = G
    program.P = P
    program.b = b
    program.c = c
    program.h = h
    # println("size of A $(size(A))")
    # println("size of G $(size(G))")
    # println("size of P $(size(P))")
    # println("size of b $(length(b))")
    # println("size of c $(length(c))")
    # println("size of h $(length(h))")
    return program
end

export build_program