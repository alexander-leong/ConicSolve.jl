#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

mutable struct ConeConstraint
    cone::Cone
    lhs::AbstractArray{Float64}
    rhs::Union{AbstractArray{Float64}, Float64}
end

mutable struct ConeQP_IR
    obj::Vector{Float64}
    quad_obj::Matrix{Float64}
    affine_constraints::Vector{ConeConstraint}
    inequality_constraints::Vector{ConeConstraint}

    ids_cones::Vector{UInt64}
    num_vars::UInt64
    function ConeQP_IR()
        ir = new()
        ir.obj = []
        ir.quad_obj = Matrix{Float64}(undef, 0, 0)
        ir.affine_constraints = []
        ir.inequality_constraints = []
        return ir
    end
end

"""
    ConeQP

Represents a Conic Quadratic Program.
"""
mutable struct ConeQP
    A::Union{AbstractArray{Float64}, Nothing}
    G::Union{AbstractArray{Float64}, Nothing}
    KKT_b::AbstractVector{Float64}
    KKT_x::AbstractVector{Float64}
    P::Union{AbstractArray{Float64}, Nothing}
    b::Union{AbstractVector{Float64}, Nothing}
    c::Union{AbstractVector{Float64}, Nothing}
    h::Union{AbstractVector{Float64}, Nothing}
    s::AbstractVector{Float64}
    z::AbstractVector{Float64}
    α::Float64
    
    cones::Vector{Cone}
    cones_inds::Vector{Int}
    cones_p::Vector{Int}

    inds_c::UnitRange{Int64}
    inds_b::UnitRange{Int64}
    inds_h::UnitRange{Int64}

    is_feasibility_problem::Bool
    kktsystem::KKTSystem

    program_ir::ConeQP_IR

    """
        ConeQP(A, G, P, b, c, h, cones)
    
    Constructs a new Conic Quadratic Program of the form:
    ```math
    \\begin{aligned}
    \\text{minimize}\\qquad &
    (1/2)x^TPx + c^Tx \\\\
    \\text{subject to}\\qquad &
    Gx + s = h \\\\
    & Ax = b \\\\
    & s \\succeq 0
    \\end{aligned}
    ```
    
    # Parameters:
    * `A`: The block matrix A in the KKT matrix
    * `G`: The block matrix G in the KKT matrix
    * `P`: The block matrix P in the KKT matrix
    * `b`: The vector b corresponding to ``Ax = b``
    * `c`: The vector c corresponding to ``c^Tx``
    * `h`: The vector h corresponding to ``Gx + s = h``
    * `cones`: A vector of cone types

    The cones vector is an ordered vector corresponding to the
    conic constraints defined by:
    ```math
    \\begin{aligned}
    Gx + s = h \\\\
    s \\succeq_K 0
    \\end{aligned}
    ```
    where ``\\succeq_K`` is a generalized inequality with respect to cone K.

    NOTE: The K is sometimes dropped to simplify notation.
    """
    function ConeQP(A::Union{AbstractArray{Float64}, Nothing},
                    G::Union{AbstractArray{Float64}, Nothing},
                    P::Union{AbstractArray{Float64}, Nothing},
                    b::Union{AbstractArray{Float64}, Nothing},
                    c::Union{AbstractArray{Float64}, Nothing},
                    h::Union{AbstractArray{Float64}, Nothing},
                    cones::Vector{Cone})
        cone_qp = new()
        cone_qp.program_ir = ConeQP_IR()
        cone_qp.A = A
        cone_qp.G = G
        cone_qp.P = P
        cone_qp.b = b
        cone_qp.c = c
        cone_qp.h = h
        cone_qp.cones = cones
        return cone_qp
    end
    
    """
        ConeQP(G, P, c, h, cones)
    
    Constructs a new Conic Quadratic Program of the form:
    ```math
    \\begin{aligned}
    \\text{minimize}\\qquad &
    (1/2)x^TPx + c^Tx \\\\
    \\text{subject to}\\qquad &
    Gx + s = h \\\\
    & s \\succeq 0
    \\end{aligned}
    ```
    
    # Parameters:
    * `G`: The block matrix G in the KKT matrix
    * `P`: The block matrix P in the KKT matrix
    * `c`: The vector c corresponding to ``c^Tx``
    * `h`: The vector h corresponding to ``Gx + s = h``
    * `cones`: A vector of cone types
    """
    function ConeQP(G::AbstractArray{Float64},
                    P::AbstractArray{Float64},
                    c::AbstractArray{Float64},
                    h::AbstractArray{Float64},
                    cones::Vector{Cone})
        cone_qp = ConeQP(undef, G, P, undef, c, h, cones)
        return cone_qp
    end

    function ConeQP()
        cone_qp = new()
        cone_qp.program_ir = ConeQP_IR()
        cone_qp.A = nothing
        cone_qp.G = nothing
        cone_qp.P = nothing
        cone_qp.b = []
        cone_qp.c = []
        cone_qp.h = []
        cone_qp.cones = []
        return cone_qp
    end
end

function find_constraints(constraints, predicate)
    match_constraints = []
    for (i, constraint) in enumerate(constraints)
        if predicate(constraint) == true
            push!(match_constraints, (i, constraint))
        end
    end
    return match_constraints
end

function find_affine_constraints(program::ConeQP, predicate)
    ir = program.program_ir
    return find_constraints(ir.affine_constraints, predicate)
end

function find_inequality_constraints(program::ConeQP, predicate)
    ir = program.program_ir
    return find_constraints(ir.inequality_constraints, predicate)
end

export find_affine_constraints
export find_inequality_constraints

function remove_affine_constraints_by_indices(program::ConeQP, indices)
    ir = program.program_ir
    ir.affine_constraints = ir.affine_constraints[indices]
    program.A = program.A[indices, :]
    program.b = program.b[indices]
end

function remove_inequality_constraints_by_indices(program::ConeQP, indices)
    ir = program.program_ir
    ir.inequality_constraints = ir.inequality_constraints[indices]
    program.G = program.G[indices, :]
    program.h = program.h[indices]
end

export remove_affine_constraints_by_indices
export remove_inequality_constraints_by_indices

function get_constraint(program::ConeQP, cone::Cone, v::Vector{Float64})
    ir = program.program_ir
    k = findfirst(id -> id == objectid(cone), ir.ids_cones)
    inds = program.cones_inds[k]+1:program.cones_inds[k+1]
    return v[inds]
end

function get_constraint(program::ConeQP, constraint::ConeConstraint, v::Vector{Float64})
    ir = program.program_ir
    cone = constraint.cone
    k = findfirst(id -> id == objectid(cone), ir.ids_cones)
    inds = program.cones_inds[k]+1:program.cones_inds[k+1]
    v[inds] = constraint.lhs
    return v
end

export get_constraint

function get_constraint_matrix(program::ConeQP, constraints::Vector{ConeConstraint})
    n = program.cones_inds[end]
    m = length(constraints)
    V = zeros((m, n))
    for (i, constraint) in enumerate(constraints)
        v = zeros(n)
        v = get_constraint(program, constraint, v)
        V[i, :] = v
    end
    return V
end

function get_affine_constraint_matrix(program::ConeQP)
    ir = program.program_ir
    A = get_constraint_matrix(program, ir.affine_constraints)
    b = [constraint.rhs for constraint in ir.affine_constraints]
    return A, b
end

function get_inequality_constraint_matrix(program::ConeQP)
    ir = program.program_ir
    n = length(program.c)
    G = -Matrix{Float64}(I, n, n)
    h = zeros(n)
    if length(ir.inequality_constraints) > 0
        G = vcat(G, get_constraint_matrix(program, ir.inequality_constraints))
        h = vcat(h, [constraint.rhs for constraint in ir.inequality_constraints])
    end
    return G, h
end

function set_objective(program::ConeQP, P::Matrix{Float64}, c::Vector{Float64}=[])
    ir = program.program_ir
    ir.quad_obj = P
    if c == []
        n = size(P, 1)
        c = zeros(n)
    end
    ir.obj = c
end

function set_objective(program::ConeQP, c::Vector{Float64})
    ir = program.program_ir
    n = length(c)
    ir.quad_obj = zeros((n, n))
    ir.obj = c
end

function build_program(program::ConeQP)
    set_cones_inds(program)
    
    ir = program.program_ir
    ir.ids_cones = [objectid(cone) for cone in program.cones]
    P = ir.quad_obj
    c = ir.obj
    program.P = P
    program.c = c

    A, b = get_affine_constraint_matrix(program)
    G, h = get_inequality_constraint_matrix(program)
    program.A = A
    program.G = G
    program.b = b
    program.h = h
    return program
end

export build_program

function add_variable(program::ConeQP, cone::Cone, p::Int64)
    cone.p = p
    push!(program.cones, cone)
    return cone
end

function add_affine_constraint(program::ConeQP, cone::Cone, lhs::AbstractArray{Float64}, rhs::Union{AbstractArray{Float64}, Float64})
    ir = program.program_ir
    push!(ir.affine_constraints, ConeConstraint(cone, lhs, rhs))
end

function add_inequality_constraint(program::ConeQP, cone::Cone, lhs::AbstractArray{Float64}, rhs::Union{AbstractArray{Float64}, Float64})
    ir = program.program_ir
    push!(ir.inequality_constraints, ConeConstraint(cone, lhs, rhs))
end

export add_variable
export add_affine_constraint
export add_inequality_constraint

function check_program(cone_qp::ConeQP)
    A = cone_qp.A
    G = cone_qp.G
    P = cone_qp.P
    b = cone_qp.b
    c = cone_qp.c
    h = cone_qp.h
    if A != undef && !isnothing(A)
        if size(A)[2] != size(G)[2]
            throw(DimensionMismatch("Number of columns of A does not equal G"))
        end
    end
    if P != undef && !isnothing(P)
        if size(G)[2] != size(P)[2]
            throw(DimensionMismatch("Number of columns of G does not equal P"))
        end
        if size(P)[1] != size(P)[2]
            throw(DimensionMismatch("P is not square"))
        end
    end
    cone_qp.inds_c = 1:size(G)[2]
    if b != undef && !isnothing(b)
        if size(A)[1] != length(b)
            throw(DimensionMismatch("Number of rows of A does not equal b"))
        elseif size(A)[2] != length(c)
            throw(DimensionMismatch("Number of columns of A does not equal c"))
        else
            cone_qp.inds_b = cone_qp.inds_c[end]+1:cone_qp.inds_c[end]+size(A)[1]
            cone_qp.inds_h = cone_qp.inds_b[end]+1:cone_qp.inds_b[end]+size(G)[1]
        end
    else
        cone_qp.inds_h = cone_qp.inds_c[end]+1:cone_qp.inds_c[end]+size(G)[1]
    end
    if size(G)[1] != length(h)
        throw(DimensionMismatch("Number of rows of G does not equal h"))
    end
    cone_qp.is_feasibility_problem = (P != undef && !isnothing(P) && P != zeros(size(P)) || c != zeros(size(G)[2]))
end

function get_variable_primal(program::ConeQP)
    x = program.KKT_x[program.inds_c]
    s = program.s
    return (x, s)
end

function get_constraint_dual(program::ConeQP)
    y = program.KKT_x[program.inds_b]
    z = program.KKT_x[program.inds_h]
    return (y, z)
end

function set_cones_inds(program::ConeQP)
    program.cones_inds = [0]
    println("set cones inds")
    for (_, cone) in enumerate(program.cones)
        ind = program.cones_inds[end] + get_size(cone)
        # println(ind)
        push!(program.cones_inds, ind)
    end
end