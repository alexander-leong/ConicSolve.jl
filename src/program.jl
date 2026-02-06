#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra
using OperatorScaling

"""
    Program API

The program API has been developed to address critical pain points in the modelling and development of solver algorithms
- Symmetric reduction for polynomial optimization problems (sos_sg.jl)
- Face reduction (ConicSolveFR.jl)
and for defining general purpose cone programs.

This is achieved by significantly reducing the need to index into matrix/vector structures so that users 
have more cognitive capacity to spend on the things that matter, solving the right problems the right way!
"""

mutable struct ObjectiveFunction
    c::Vector{Vector{Float64}}
    cones::Vector{Cone}

    function ObjectiveFunction()
        obj = new()
        obj.c = []
        obj.cones = []
        return obj
    end
end

export ObjectiveFunction

function minimize(args...)
    obj = ObjectiveFunction()
    for arg in args
        push!(obj.c, arg.lhs)
        push!(obj.cones, arg.cone)
    end
    return obj
end

export minimize

"""
    PrimalObjective

Represents a primal objective quadratic function
```math
x^TPx + cx
```
or primal objective linear function
```math
cx
```
for the variable ``x`` that is defined over the given cone.
"""
mutable struct PrimalObjective
    cone::Cone
    P::AbstractArray{Float64}
    c::AbstractArray{Float64}
end

mutable struct ConeQP_IR
    obj::Vector{PrimalObjective}
    affine_constraints::Vector{ConicExpression{<:Cone}}
    inequality_constraints::Vector{ConicExpression{<:Cone}}
    _all_affine_constraints::Vector{ConicExpression{<:Cone}}
    _all_inequality_constraints::Vector{ConicExpression{<:Cone}}

    ids_cones::Vector{UInt64}
    num_vars::UInt64
    function ConeQP_IR()
        ir = new()
        ir.obj = []
        ir.affine_constraints = Vector{ConicExpression{<:Cone}}()
        ir.inequality_constraints = Vector{ConicExpression{<:Cone}}()
        ir._all_affine_constraints = Vector{ConicExpression{<:Cone}}()
        ir._all_inequality_constraints = Vector{ConicExpression{<:Cone}}()
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

    inv_P::Union{AbstractArray{Float64}, Nothing}

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
        if all(cone_qp.P .== 0) == false
            F = svd(cone_qp.P)
            cone_qp.inv_P = F.Vt' * diagm(inv.(F.S)) * F.U'
        end
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

export ConeQP

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
The predicate is a function that takes a ``ConicExpression`` and returns true or false.
"""
function find_affine_constraints(program::ConeQP, predicate)
    ir = program.program_ir
    return find_constraints(ir._all_affine_constraints, predicate)
end

"""
    find_inequality_constraints(program, predicate)

Returns a list of inequality constraints in the given program that satisfy the predicate.
The predicate is a function that takes a ``ConicExpression`` and returns true or false.
"""
function find_inequality_constraints(program::ConeQP, predicate)
    ir = program.program_ir
    return find_constraints(ir._all_inequality_constraints, predicate)
end

"""
    find_affine_constraints(program, cone)

Returns a list of affine constraints in the given program that reference the given cone object.
"""
function find_affine_constraints_by_cone(program::ConeQP, cone::Cone)
    ir = program.program_ir
    return [constraint for constraint in ir._all_affine_constraints if objectid(constraint.cone) == objectid(cone)]
end

"""
    find_inequality_constraints(program, cone)

Returns a list of inequality constraints in the given program that reference the given cone object.
"""
function find_inequality_constraints_by_cone(program::ConeQP, cone::Cone)
    ir = program.program_ir
    return [constraint for constraint in ir._all_inequality_constraints if objectid(constraint.cone) == objectid(cone)]
end

export find_affine_constraints
export find_inequality_constraints
export find_affine_constraints_by_cone
export find_inequality_constraints_by_cone

function remove_affine_constraints_by_indices(program::ConeQP, indices)
    ir = program.program_ir
    ir._all_affine_constraints = ir._all_affine_constraints[indices]
    program.A = program.A[indices, :]
    program.b = program.b[indices]
end

function remove_inequality_constraints_by_indices(program::ConeQP, indices)
    ir = program.program_ir
    ir._all_inequality_constraints = ir._all_inequality_constraints[indices]
    program.G = program.G[indices, :]
    program.h = program.h[indices]
end

export remove_affine_constraints_by_indices
export remove_inequality_constraints_by_indices

function get_indices_of_constraint(program::ConeQP, cone::Cone)
    ir = program.program_ir
    k = findfirst(id -> id == objectid(cone), ir.ids_cones)
    inds = program.cones_inds[k]+1:program.cones_inds[k+1]
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

function get_constraint(program::ConeQP, constraint::ConicExpression, v::Vector{Float64})
    inds = get_indices_of_constraint(program, constraint.cone)
    v[inds] = constraint.lhs
    return v
end

function get_constraint(program::ConeQP, constraint::ConicExpression, v::Matrix{Float64})
    inds = get_indices_of_constraint(program, constraint.cone)
    v[:, inds[constraint.inds]] = constraint.lhs
    return v
end

function get_constraint_indices(program::ConeQP, constraints::Vector{ConicExpression})
    n = program.cones_inds[end]
    function num_constraints(constraint)
        if length(size(constraint.lhs)) == 1
            return 1
        end
        return size(constraint.lhs, 1)
    end
    num_rows = [num_constraints(constraint) for constraint in constraints]
    inds = [0, cumsum(num_rows)...]
    m = sum(num_rows)
    return m, n, inds, num_rows
end

export get_constraint

function get_constraint_matrix(program::ConeQP, constraints::Vector{ConicExpression{<:Cone}}, V=[])
    function set_link_constraints!(program::ConeQP, constraint::ConicExpression, V)
        for link_constraint in constraint.link_constraints
            inds = get_indices_of_constraint(program, link_constraint.cone)
            V[:, inds] = link_constraint.lhs
        end
    end
    m, n, inds, num_rows = get_constraint_indices(program, constraints)
    if V == []
        V = zeros((m, n))
    end
    for (i, constraint) in enumerate(constraints)
        if num_rows[i] == 1 # TODO refactor
            v = get_constraint(program, constraint, zeros(n))
            V[i, :] = v
            set_link_constraints!(program, constraint, V[i, :])
        else
            v = get_constraint(program, constraint, zeros((num_rows[i], n)))
            V[inds[i]+1:inds[i+1], :] = v
            set_link_constraints!(program, constraint, V[inds[i]+1:inds[i+1], :])
        end
    end
    return V
end

export get_constraint_matrix

function get_affine_constraint_matrix(program::ConeQP, allocated=false)
    ir = program.program_ir
    v = allocated == false ? [] : program.A
    A = get_constraint_matrix(program, ir._all_affine_constraints, v)
    b = vcat([constraint.rhs for constraint in ir._all_affine_constraints]...)
    return A, b
end

function get_inequality_constraint_matrix(program::ConeQP, allocated=false)
    ir = program.program_ir
    for cone in program.cones
        n = get_size(cone)
        G = -Matrix{Float64}(I, n, n)
        h = zeros(n)
        add_inequality_constraint(program, cone, G, h)
    end
    G = get_constraint_matrix(program, ir._all_inequality_constraints)
    h = vcat([constraint.rhs for constraint in ir._all_inequality_constraints]...)
    return G, h
end

function get_primal_objective(program::ConeQP)
    ir = program.program_ir
    n = program.cones_inds[end]
    P = zeros((n, n))
    c = zeros(n)
    for obj in ir.obj
        cone = obj.cone
        k = findfirst(id -> id == objectid(cone), ir.ids_cones)
        inds = program.cones_inds[k]+1:program.cones_inds[k+1]
        P[inds, inds] = obj.P
        c[inds] = obj.c
    end
    return P, c
end

"""
    set_objective(program, cone, P, c)

Set the primal objective quadratic function
"""
function set_objective(program::ConeQP, cone::Cone, P::Matrix{Float64}, c::Vector{Float64}=[])
    ir = program.program_ir
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
function set_objective(program::ConeQP, cone::Cone, c::Vector{Float64})
    ir = program.program_ir
    n = length(c)
    P = zeros((n, n))
    primal_obj = PrimalObjective(cone, P, c)
    push!(ir.obj, primal_obj)
end

export set_objective

"""
    add_variable(program, cone, p)

Add cone variable of size p to given program
"""
function add_variable(program::ConeQP, cone::Cone, p::Int64)
    cone.p = p
    if objectid(cone) in program.cones
        return cone
    end
    push!(program.cones, cone)
    return cone
end

export add_variable

function update_program(program::ConeQP)
    set_cones_inds(program)
    ir = program.program_ir
    program.P = ir.quad_obj
    program.c = ir.obj
    program.A, program.b = get_affine_constraint_matrix(program, true)
    program.G, program.h = get_inequality_constraint_matrix(program, true)
    program.A = @view program.A[:, 1:program.cones_inds[end]]
    program.G = @view program.G[:, 1:program.cones_inds[end]]
    return program
end

export update_program

"""
    add_to_affine_constraint(constraint, cone, lhs)

Sets the elements of the left hand side, *lhs* of an existing (affine or inequality) constraint with respect to a different cone.
Used to define a constraint that is "jointly linked" between variables in constraint.cone and cone.

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

"""
    constrain_to_cone(program, constraint, cone, lhs, rhs)

Sets an additional constraint given by *cone*, *lhs*, *rhs* on the same variable *constraint.cone*

Implements
``K₁ ∩ K₂ ∩ ... Kₙ``
"""
function constrain_to_cone(program::ConeQP, constraint::ConicExpression, intersecting_constraint::ConicExpression)
    push!(constraint.intersecting_constraints, intersecting_constraint)
    ir = program.program_ir
    push!(ir._all_inequality_constraints, intersecting_constraint)
    return constraint.cone
end

"""
    add_affine_constraint(program, cone, lhs, rhs)

Add to the program affine constraint ``lhs * x = rhs`` with respect to the cone
"""
function add_affine_constraint(program::ConeQP, cone::Cone, lhs::AbstractArray{Float64}, rhs::Union{AbstractArray{Float64}, Float64})
    ir = program.program_ir
    constraint = ConicExpression(cone, lhs, rhs)
    push!(ir.affine_constraints, constraint)
    push!(ir._all_affine_constraints, constraint)
    return ir.affine_constraints[end]
end

export add_affine_constraint

function set_affine_constraint(constraint::ConicExpression, cone::Cone, lhs::AbstractArray{Float64})
    add_to_affine_constraint(constraint, cone, lhs)
end

"""
    add_inequality_constraint(program, cone, lhs, rhs)

Add to the program inequality constraint ``lhs * x ≤ rhs`` with respect to the cone
"""
function add_inequality_constraint(program::ConeQP, cone::Cone, lhs::AbstractArray{Float64}, rhs::Union{AbstractArray{Float64}, Float64})
    ir = program.program_ir
    constraint = ConicExpression(cone, lhs, rhs)
    push!(ir.inequality_constraints, constraint)
    push!(ir._all_inequality_constraints, constraint)
    return ir.inequality_constraints[end]
end

function set_inequality_constraint(constraint::ConicExpression, cone::Cone, lhs::AbstractArray{Float64})
    add_to_affine_constraint(constraint, cone, lhs)
end

export add_variable
export add_affine_constraint
export add_inequality_constraint

export set_affine_constraint
export set_inequality_constraint

function set_cones_inds(program::ConeQP)
    program.cones_inds = [0]
    for (_, cone) in enumerate(program.cones)
        ind = program.cones_inds[end] + get_size(cone)
        push!(program.cones_inds, ind)
    end
end

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
    if size(G)[2] != length(c)
        throw(DimensionMismatch("Number of columns of G does not equal c"))
    end
    if b != undef && !isnothing(b)
        if size(A)[1] != length(b)
            throw(DimensionMismatch("Number of rows of A does not equal b"))
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
    if cone_qp.cones_inds == []
        set_cones_inds(cone_qp)
    end
    if size(G, 1) != cone_qp.cones_inds[end]
        throw(DimensionMismatch("Number of rows of G does not equal total size of cones"))
    end
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

export get_constraint_dual

function get_num_constraints(program::ConeQP)
    num_constraints = 0
    if !isnothing(program.A)
        num_constraints += size(program.A)[1]
    end
    num_constraints += size(program.G)[1]
    return num_constraints
end

function check_preconditions(qp::ConeQP)
    if !isnothing(qp.A) && rank(qp.A) < size(qp.A)[1]
        @error "Values of A are inconsistent or redundant."
        @assert false
    end
    if !isnothing(qp.A)
        if rank([qp.P qp.A' qp.G']) < length(qp.c)
            @error "There are some constraints in the problem that are either
            redundant or inconsistent."
            @assert false
        end
    else
        if rank([qp.P qp.G']) < length(qp.c)
            @error "There are some constraints in the problem that are either
            redundant or inconsistent."
            @assert false
        end
    end
end

function update_affine_constraints(program::ConeQP)
    program.kktsystem.A = program.A
    program.kktsystem.b = program.b
end

function update_inequality_constraints(program::ConeQP)
    program.kktsystem.G = program.G
    program.kktsystem.h = program.h
end

export update_affine_constraints
export update_inequality_constraints

function apply_equilibration(program::ConeQP, max_iter::Int=100)
    @info "Performing Ruiz Equilibration"
    A_scaled, D1, D2 = equilibrate(program.A, max_iter=max_iter)
    b_scaled = D1 * program.b
    program.A = A_scaled
    program.b = b_scaled
    return D2
end

# rank revealing qr, remove redundant constraints
function rrqr(program::ConeQP, tol=1e-9)
    A = program.A
    F = qr(A', ColumnNorm())
    # get list of row indices where diagonal elements of R satisfy tol
    inds = [i for (i, v) in enumerate(diag(F.R)) if abs(v) >= tol]
    inds = [x[2] for x in findall(val -> val == 1, F.P'[inds, :])]
    return inds
end

export rrqr

function apply_regularization(program::ConeQP, tol=1e-9)
    @info "Performing RRQR regularization"
    inds = rrqr(program, tol)
    remove_affine_constraints_by_indices(program, inds)
end

export apply_regularization

function print_graphic()
    fp = open("graphic.txt", "r")
    words = readlines(fp, keep=true)
    for word in words
        print(word)
    end
end

function print_header()
    # print_graphic()
    println("Alexander Leong, 2025")
    println("Version 0.0.3")
    println(repeat("-", 144))
end
