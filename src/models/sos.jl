#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using LinearAlgebra

"""
    SOS

Represents a Sum of Squares Optimization (SOS) Program. \\
The univariate polynomial ``p(x) = a_nx^n + a_{n-1}x^{n-1} + ... + a_{1}x^1 + a_0``
is represented in quadratic form as:
``p(x) = v^TQv`` where:
```math
v = \\begin{bmatrix}
    x^n \\\\ x^{n-1} \\\\ ... \\\\ x^{1} \\\\ 1
\\end{bmatrix}
```
Multivariate polynomials and other monomial basis are not supported at this stage.
"""
mutable struct SOS
    A
    G
    P
    b
    c
    h

    n # order of polynomial
    p_inds # indices of each monomial of order n in A
    diag_inds
    cones::Vector{Cone}

    @doc"""
        SOS(n)
    
    Constructs an SOS optimization problem. \\
    
    # Parameters:
    * `n`: The order of the polynomial
    """
    function SOS(n)
        sos = new()
        sos.cones = []
        sos.p_inds = []
        push!(sos.cones, PSDCone(n))
        sos.n = n
        sos.diag_inds = get_diagonal_idx(n)
        d = sos.n - 1
        A_size = Int(((d + 1) * (d + 2)) / 2)
        # set SOS constraints
        sos.A = get_A_matrix(sos)
        num_vars = size(sos.A)[2]
        G = -Matrix(1.0I, num_vars, num_vars)
        sos.G = G
        sos.b = zeros(size(sos.A)[1])
        sos.h = zeros(size(sos.G)[1])
        return sos
    end
end

function get_a_row(i, mask)
    for u = size(mask)[1]-i:size(mask)[1]
        mask[u, i + 1 - (size(mask)[1] - u)] = 1.0
    end
    val = mat_to_lower_vec(mask)
    return val
end

function get_A_matrix(sos::SOS)
    d = sos.n - 1
    A_size = Int(((d + 1) * (d + 2)) / 2)
    A = zeros((d+1, A_size))
    A = hcat(A, -I(d+1))
    # get mask of V'V (V for Vandermonde) matrix to set SOS constraint
    mask = [zeros(Float64, (d+1, d+1)) for _ = 1:d]
    rows = 2:size(A)[1]
    cols = 1:A_size
    push!(sos.p_inds, A_size)
    # set constraints for monomial terms of order 1 and above
    for (k, v) in enumerate(mask)
        num_inds = size(v)[1] - (size(v)[1] - k - 1)
        push!(sos.p_inds, sos.p_inds[end] - num_inds)
        A[rows[k], cols] = get_a_row(k, v)
    end
    reverse!(sos.p_inds)
    # set constraint for the constant term in the polynomial (order 0)
    A[1, d+1] = 1
    return A
end

function get_inds_2d(sos::SOS, n)
    m = Int(floor(n / 2)) + 1
    n = n - m + 1
    i = sos.n - m + 1
    inds = []
    for j = sos.n - n + 1:-1:1
        if j > i
            i = i + 1
            continue
        end
        push!(inds, CartesianIndex(i, j))
        i = i + 1
        if i > sos.n || j == 1
            break
        end
    end
    return inds
end

function get_inds(sos::SOS, n)
    inds = get_inds_2d(sos, n)
    inds = lower_triangular_from_2d_idx(sos.n, inds)
    inds = map(x -> CartesianIndex(1, x), inds)
    return inds
end

"""
    set_objective(sos, c)

Set the objective function of the SOS, which is the function
``\\langle c, x \\rangle``
"""
function set_objective(sos::SOS, c)
    num_vars = size(sos.A)[2]
    if !isdefined(sos, :P)
        sos.P = zeros((num_vars, num_vars))
    end
    sos.c = zeros(size(sos.A)[2])
    sos.c[1:length(c)] = c
    # Nonnegative orthant follows from PSD cone constraint
    p = size(sos.G)[1] - sos.diag_inds[end]
    if p > 0
        push!(sos.cones, NonNegativeOrthant(p))
    end
end

"""
    set_objective(sos, P, c)

Set the objective function of the SOS
"""
function set_objective(sos::SOS, P, c)
    sos.P = P
    set_objective(sos, c)
end

"""
    set_diagonal_Q_constraint(sos)

Set the off diagonal entries of Q to zero.
"""
# FIXME remove?
function set_diagonal_Q_constraint(sos::SOS)
    idx = get_off_diagonal_idx(sos.n)
    idx = map(x -> CartesianIndex(x[1], x[2]), enumerate(idx))
    d = sos.n - 1
    A_size = Int(((d + 1) * (d + 2)) / 2)
    A = Matrix(1.0I, length(idx), A_size)
    b = zeros(length(idx))
    A[idx] .= 1
    if isdefined(sos, :A)
        sos.A = vcat(sos.A, A)
        sos.b = vcat(sos.b, b)
    else
        sos.A = A
        sos.b = b
    end
end

"""
    add_polynomial_equality_constraint(sos, b, p)

Set an equality constraint on the polynomial to equal the value b.
"""
function add_polynomial_equality_constraint(sos::SOS, b, p)
    A = zeros((1, size(sos.A)[2]))
    for i = eachindex(p)
        i_inds = get_inds(sos, i)
        A[i_inds] .= p[i]
    end
    sos.A = vcat(sos.A, A)
    sos.b = vcat(sos.b, b)
end

"""
    add_polynomial_inequality_constraint(sos, h, p)

Set an inequality constraint on the polynomial to equal the value h.
"""
function add_polynomial_inequality_constraint(sos::SOS, h, p)
    G = zeros((1, size(sos.G)[2]))
    for i = eachindex(p)
        i_inds = get_inds(sos, i)
        G[i_inds] .= p[i]
    end
    sos.G = vcat(sos.G, G)
    sos.h = vcat(sos.h, h)
end

"""
    get_qp(sos)

Get the Cone QP object representing the SOS optimization problem
"""
function get_qp(sos::SOS)
    cone_qp = ConeQP(sos.A, sos.G, sos.P, sos.b, sos.c, sos.h, sos.cones)
    return cone_qp
end

export SOS
export get_inds_2d
export set_objective
export set_diagonal_Q_constraint
export add_polynomial_equality_constraint
export add_polynomial_inequality_constraint
export get_qp