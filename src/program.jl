#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using DynamicPolynomials
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

    function PrimalObjective(cone, P, c)
        obj = new()
        obj.cone = cone
        obj.P = P
        obj.c = c
        return obj
    end
end

mutable struct ConicVariables
    cones::Vector{Cone}
    cones_inds::Vector{Int}
    cones_p::Vector{Int}
    offset::Int
    function ConicVariables()
        obj = new()
        obj.cones = []
        obj.cones_inds = []
        obj.cones_p = []
        obj.offset = 0
        return obj
    end
end

function get_size(variables::ConicVariables)
    return variables.cones_inds[end]
end

function get_inds(variables::ConicVariables, i::Int)
    return variables.cones_inds[i]+1:variables.cones_inds[i+1]
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
    
    vars::ConicVariables
    aux_vars::ConicVariables

    inds_c::UnitRange{Int64}
    inds_b::UnitRange{Int64}
    inds_h::UnitRange{Int64}
    inds_solution::UnitRange{Int64}

    is_feasibility_problem::Bool
    kktsystem::KKTSystem

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
        cone_qp.vars = ConicVariables()
        cone_qp.vars.cones = cones
        cone_qp.aux_vars = ConicVariables()
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
        cone_qp.A = nothing
        cone_qp.G = nothing
        cone_qp.P = nothing
        cone_qp.b = []
        cone_qp.c = []
        cone_qp.h = []
        cone_qp.vars = ConicVariables()
        cone_qp.aux_vars = ConicVariables()
        return cone_qp
    end
end

export ConeQP

function find_cone_from_id(program::ConeQP, id::UInt64)
    idx = findfirst(cone -> id == objectid(cone), program.vars.cones)
    return program.vars.cones[idx]
end

function get_size(program::ConeQP, ids_cones::Vector{UInt64})
    if length(ids_cones) == 0
        return 0
    end
    return sum([get_size(find_cone_from_id(program, id)) for id in ids_cones])
end

"""
    add_variable(program, cone, p)

Add cone variable of size p to given program
"""
function add_variable(program::ConeQP, cone::Cone, p::Int64)
    vars = program.vars
    cone.p = p
    if objectid(cone) in vars.cones
        return cone
    end
    push!(vars.cones, cone)
    return cone
end

export add_variable

function get_cones_inds(cones::Vector{Cone})
    cones_inds = [0]
    for (_, cone) in enumerate(cones)
        ind = cones_inds[end] + get_size(cone)
        push!(cones_inds, ind)
    end
    return cones_inds
end

function set_cones_inds(program::ConeQP)
    vars = program.vars
    vars.cones_inds = get_cones_inds(vars.cones)
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
    set_cones_inds(cone_qp)
    total_size = get_size(cone_qp.vars)
    if size(G, 1) != total_size
        throw(DimensionMismatch("Number of rows of G, $(size(G, 1)) does not equal total size of cones, $(total_size)"))
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

function remove_affine_constraints_by_indices(program::ConeQP, indices)
    program.A = program.A[indices, :]
    program.b = program.b[indices]
end

function remove_inequality_constraints_by_indices(program::ConeQP, indices)
    program.G = program.G[indices, :]
    program.h = program.h[indices]
end

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
    println("Alexander Leong, 2026")
    println("Version 0.0.3")
    println(repeat("-", 144))
end
