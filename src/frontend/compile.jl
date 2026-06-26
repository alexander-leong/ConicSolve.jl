#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

function find_constraints(constraints, predicate)
    match_constraints = []
    for (i, constraint) in enumerate(constraints)
        if predicate(constraint) == true
            push!(match_constraints, (i, constraint))
        end
    end
    return match_constraints
end

"""
    find_affine_constraints(program, predicate)

Returns a list of affine constraints in the given program that satisfy the predicate.
The predicate is a function that takes a *ConicExpression* and returns true or false.
"""
function find_affine_constraints(ir::ConeQP_IR, predicate)
    return find_constraints(ir._all_affine_constraints, predicate)
end

"""
    find_inequality_constraints(program, predicate)

Returns a list of inequality constraints in the given program that satisfy the predicate.
The predicate is a function that takes a *ConicExpression* and returns true or false.
"""
function find_inequality_constraints(ir::ConeQP_IR, predicate)
    return find_constraints(ir._all_inequality_constraints, predicate)
end

"""
    find_affine_constraints(program, cone)

Returns a list of affine constraints in the given program that reference the given cone object.
"""
function find_affine_constraints_by_cone(ir::ConeQP_IR, cone::Cone)
    return [constraint for constraint in ir._all_affine_constraints if objectid(constraint.cone) == objectid(cone)]
end

"""
    find_inequality_constraints(program, cone)

Returns a list of inequality constraints in the given program that reference the given cone object.
"""
function find_inequality_constraints_by_cone(ir::ConeQP_IR, cone::Cone)
    return [constraint for constraint in ir._all_inequality_constraints if objectid(constraint.cone) == objectid(cone)]
end

export find_affine_constraints
export find_inequality_constraints
export find_affine_constraints_by_cone
export find_inequality_constraints_by_cone

function get_size(program::ConeQP, ir::ConeQP_IR)
    if isempty(ir.obj)
        return get_size(program.vars)
    end
    n = sum([length(obj.c) for obj in ir.obj])
    return n
end

function get_indices_of_constraint(program::ConeQP, cone::Cone)
    # program = program_int.cone_qp
    # ir = program_int.ir
    vars = program.vars
    ids_cones = [objectid(cone) for cone in vars.cones]
    k = findfirst(id -> id == objectid(cone), ids_cones)
    inds = get_inds(vars, k)
    return inds
end

export get_indices_of_constraint

function get_elements_by_constraint_inds(program::ConeQP, cone::Cone, v)
    inds = get_indices_of_constraint(program, cone)
    return v[inds]
end

function get_elements_by_constraint_inds(program::ConeQP, cone::Cone, V::Matrix{Float64})
    inds = get_indices_of_constraint(program, cone)
    return V[inds, :]
end

export get_elements_by_constraint_inds

function set_column_indices!(program::ConeQP, constraint::ConicExpression, v::Vector{Float64})
    inds = get_indices_of_constraint(program, constraint.cone)
    v[inds] = constraint.lhs
    return v
end

function set_column_indices!(program::ConeQP, constraint::ConicExpression, v::Matrix{Float64})
    inds = get_indices_of_constraint(program, constraint.cone)
    if maximum(constraint.inds) > size(v, 2)
        throw(DimensionMismatch("One or more constraints do not have the correct dimensions"))
    end
    v[:, inds[constraint.inds]] = constraint.lhs
    return v
end

function set_column_indices!(program::ConeQP, constraints::Vector{ConicExpression{<:Cone}}, V)
    for constraint in constraints
        set_column_indices!(program, constraint, V)
    end
end

export set_column_indices!

function get_constraint_indices(program::ConeQP, constraints::AllConstraints, n=get_size(program.vars))
    function num_constraints(constraint::ConicExpression)
        if length(size(constraint.lhs)) == 1
            return 1
        end
        return size(constraint.lhs, 1)
    end
    function num_constraints(expression::IntersectingConstraint{T, U}) where {T<:Cone, U<:Cone}
        return num_constraints(expression.constraint)
    end
    num_rows = [num_constraints(constraint) for constraint in constraints]
    inds = [0, cumsum(num_rows)...]
    m = sum(num_rows)
    return m, n, inds, num_rows
end

function set_row_indices!(program::ConeQP, constraint::ConicExpression{<:Cone}, V, v_alloc, row_inds)
    V[row_inds, :] = set_column_indices!(program, constraint, v_alloc)
    set_column_indices!(program, constraint.link_constraints, V[row_inds, :])
end

function set_row_indices!(program::ConeQP, expression::IntersectingConstraint{<:Cone, <:Cone}, V, v_alloc, row_inds)
    inds = get_indices_of_constraint(program, expression.cone)
    constraint = expression.constraint
    v_alloc[:, inds] = constraint.lhs
    V[row_inds, :] = v_alloc
end

function get_constraint_matrix(program::ConeQP, constraints::AllConstraints, V=[], n=get_size(program.vars))
    if isempty(constraints) == true
        return []
    end
    m, n, inds, num_rows = get_constraint_indices(program, constraints, n)
    if V == []
        V = zeros((m, n))
    end
    for (i, constraint) in enumerate(constraints)
        v_alloc = num_rows[i] == 1 ? zeros(n) : zeros((num_rows[i], n))
        row_inds = num_rows[i] == 1 ? i : inds[i]+1:inds[i+1]
        set_row_indices!(program, constraint, V, v_alloc, row_inds)
    end
    return V
end

export get_constraint_matrix

function get_affine_constraint_matrix(program::ConeQP, ir::ConeQP_IR, allocated=false)
    v = allocated == false ? [] : program.A
    A = get_constraint_matrix(program, ir._all_affine_constraints, v)
    b = vcat([constraint.rhs for constraint in ir._all_affine_constraints]...)
    return A, b
end

function get_inequality_constraint_matrix(program::ConeQP, ir::ConeQP_IR, allocated=false)
    vars = program.vars
    # number of variables excludes additional slack variables
    n = get_size(vars)
    # add slack variables from intersecting constraints
    aux_vars = program.aux_vars
    append!(vars.cones, aux_vars.cones)
    set_cones_inds(program)
    G = get_constraint_matrix(program, ir._all_inequality_constraints, [], n)
    h = vcat([constraint.rhs for constraint in ir._all_inequality_constraints]...)
    return G, h
end

function get_primal_objective(program::ConeQP, ir::ConeQP_IR)
    vars = program.vars
    n = get_size(program, ir)
    P = zeros((n, n))
    c = zeros(n)
    for obj in ir.obj
        cone = obj.cone
        k = findfirst(id -> id == objectid(cone), ir.ids_cones)
        inds = get_inds(vars, k)
        P[inds, inds] = obj.P
        c[inds] = obj.c
    end
    return P, c
end

"""
    set_objective(program, cone, P, c)

Set the primal objective quadratic function
"""
function set_objective(ir::ConeQP_IR, cone::Cone, P::Matrix{Float64}, c::Vector{Float64}=[])
    if c == []
        n = size(P, 1)
        c = zeros(n)
    end
    primal_obj = PrimalObjective(cone, P, c)
    push!(ir.obj, primal_obj)
end

"""
    set_objective(program, cone, c)

Set the primal objective linear function
"""
function set_objective(ir::ConeQP_IR, cone::Cone, c::Vector{Float64})
    n = length(c)
    P = zeros((n, n))
    primal_obj = PrimalObjective(cone, P, c)
    push!(ir.obj, primal_obj)
end

function set_objective(program_int::ProgramInterface, cone::Cone, P::Matrix{Float64}, c::Vector{Float64}=[])
    set_objective(program_int.ir, cone, P, c)
end
function set_objective(program_int::ProgramInterface, cone::Cone, c::Vector{Float64})
    set_objective(program_int.ir, cone, c)
end

export set_objective

function add_slack_variable(program::ConeQP, cone::Cone, p::Int64)
    add_variable(program, cone, p)
    ir = program.program_ir
    if objectid(cone) in ir.ids_implicit_cones
        return
    end
    push!(ir.ids_implicit_cones, objectid(cone))
end

"""
    constrain_to_cone(program, constraint, cone, lhs, rhs)

Sets an additional constraint given by *cone*, *lhs*, *rhs* on the same variable *constraint.cone*

Implements
``K₁ ∩ K₂ ∩ ... Kₙ``
"""
function constrain_to_cone(ir::ConeQP_IR, constraint::ConicExpression, intersecting_constraint::ConicExpression)
    push!(constraint.intersecting_constraints, intersecting_constraint)
    push!(ir._all_inequality_constraints, intersecting_constraint)
    return constraint.cone
end

"""
    add_affine_constraint(program, cone, lhs, rhs)

Add to the program affine constraint ``lhs * x = rhs`` with respect to the cone
"""
function add_affine_constraint(ir::ConeQP_IR, cone::Cone, lhs::AbstractArray{Float64}, rhs::Union{AbstractArray{Float64}, Float64})
    constraint = ConicExpression(cone, lhs, rhs)
    push!(ir._all_affine_constraints, constraint)
end

export add_affine_constraint

function set_affine_constraint(constraint::ConicExpression, cone::Cone, lhs::AbstractArray{Float64})
    add_to_affine_constraint(constraint, cone, lhs)
end

"""
    add_inequality_constraint(program, cone, lhs, rhs)

Add to the program inequality constraint ``lhs * x ≤ rhs`` with respect to the cone
"""
function add_inequality_constraint(ir::ConeQP_IR, cone::Cone, lhs::AbstractArray{Float64}, rhs::Union{AbstractArray{Float64}, Float64})
    constraint = ConicExpression(cone, lhs, rhs)
    push!(ir._all_inequality_constraints, constraint)
end

function set_inequality_constraint(constraint::ConicExpression, cone::Cone, lhs::AbstractArray{Float64})
    add_to_affine_constraint(constraint, cone, lhs)
end

export add_affine_constraint
export add_inequality_constraint

export set_affine_constraint
export set_inequality_constraint

function parse_arg(program_int::ProgramInterface, arg::Cone)
    add_variable(program_int.cone_qp, arg, arg.p)
    return program_int
end

function get_default_inequality_constraint(cone::Cone)
    n = get_size(cone)
    G = -Matrix{Float64}(I, n, n)
    h = zeros(n)
    return G, h
end

function add_default_inequality_constraint(ir::ConeQP_IR, cone::Cone)
    G, h = get_default_inequality_constraint(cone)
    add_inequality_constraint(ir, cone, G, h)
end

export add_default_inequality_constraint

function define_program(program::ConeQP, obj::ObjectiveFunction, args...)
    program_int = ProgramInterface(program)
    ir = program_int.ir
    cones = program.vars.cones
    implicit_equality_constraints = []
    program_args = cones, implicit_equality_constraints
    out_program = program_int.cone_qp
    for arg in args
        out_program = dispatch(program_int, arg, program_args...)
    end
    for arg in implicit_equality_constraints
        # TODO
        out_program = parse_arg(program_int, arg)
    end
    ir.num_slack_vars = get_size(program, ir.ids_implicit_cones)
    for obj_term in obj.args
        out_program = parse_obj_arg(out_program, obj_term)
    end
    vars = program.vars
    for cone in vars.cones
        add_default_inequality_constraint(ir, cone)
    end
    return out_program
end

function define_program(program::ConeQP, args...)
    obj = ObjectiveFunction([])
    return define_program(program, obj, args...)
end

export define_program

function build_program(program_int::ProgramInterface, allocated=false)
    program = program_int.cone_qp
    ir = program_int.ir
    vars = program.vars
    if isempty(ir.obj)
        for cone in vars.cones
            set_objective(ir, cone, zeros(Float64, get_size(cone)))
        end
    end

    set_cones_inds(program)
    
    ir.ids_cones = [objectid(cone) for cone in vars.cones]
    P, c = get_primal_objective(program, ir)

    A, b = get_affine_constraint_matrix(program, ir)
    G, h = get_inequality_constraint_matrix(program, ir, allocated)
    
    program.A = A
    program.G = G
    program.P = P
    program.b = b
    program.c = c
    program.h = h
    return program
end

function build_program(program::SymmetryReducedConeQP, allocated=false)
    build_program(program.program_int, allocated)
    return program
end

export build_program