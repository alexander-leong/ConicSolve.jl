#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

include("interface.jl")
include("../program.jl")

function parse_arg(program::ConeQP, arg::ConeEqualityConstraint)
    ir = program.program_ir
    push!(ir.affine_constraints, arg.v)
    push!(ir._all_affine_constraints, arg.v)
end

function parse_arg(program::ConeQP, arg::ConeInequalityConstraint)
    ir = program.program_ir
    push!(ir.inequality_constraints, arg.v)
    push!(ir._all_inequality_constraints, arg.v)
end

function parse_arg(program::ConeQP, arg::Cone)
    add_variable(program, arg, arg.p)
end

# function parse_arg(program::ConeQP, arg::Vector{Cone})
#     ir = program.program_ir
#     cone = arg[1]
#     for cone in arg[2:end]
#         intersecting_constraint = ConicExpression(cone, Matrix{Float64}(I, cone.p, cone.p), zeros(get_size(cone)))
#         push!(ir._all_inequality_constraints, intersecting_constraint)
#     end
# end

function define_program(program::ConeQP, obj::ObjectiveFunction, args...)
    obj_terms = zip(obj.c, obj.cones)
    for obj_term in obj_terms
        set_objective(program, obj_term[2], obj_term[1])
    end
    for (i, arg) in enumerate(args)
        parse_arg(program, arg)
        # FIXME, throw exception upon invalid expression
        # else
        #     throw(ArgumentError("Invalid argument type $(typeof(arg)) at position $(i), must be either a [Cone, ConicExpression] type"))
        # end
    end
end

function define_program(program::ConeQP, args...)
    obj = ObjectiveFunction()
    for cone in program.cones
        push!(obj.c, zeros(Float64, get_size(cone)))
    end
    define_program(program, obj, args...)
end

export define_program

function build_program(program::ConeQP, allocated=false)
    set_cones_inds(program)
    
    ir = program.program_ir
    ir.ids_cones = [objectid(cone) for cone in program.cones]
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
    return program
end

export build_program