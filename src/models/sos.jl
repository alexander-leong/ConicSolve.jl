#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using DataStructures
using DynamicPolynomials
using LinearAlgebra
using MultivariateBases

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
    P
    b
    c

    n # order of sos problem
    p_inds # indices of each monomial of order n in A
    diag_inds
    cones::Vector{Cone}

    char_map

    """
        SOS(n)
    
    Constructs an SOS optimization problem. \\
    
    # Parameters:
    * `n`: The order of the polynomial
    """
    function SOS(n::Int, variables)
        sos = new()
        sos.cones = []
        sos.p_inds = []
        push!(sos.cones, PSDCone(n))
        sos.n = n
        sos.diag_inds = get_diagonal_idx(n)
        # set SOS constraints
        sos.A = []
        sos.b = []
        
        pbasis = collect(maxdegree_basis(MonomialBasis, variables, n-1))
        pmap2d = pbasis * pbasis'
        sos.char_map = get_vec_inds_map(pmap2d)
        return sos
    end
end

mutable struct SOSPolynomial
    p::DynamicPolynomials.Polynomial
    sos::SOS
end

export SOSPolynomial

"""
    get_inds_2d(sos, n)

Gets a list of lower triangular indices of the coefficient terms corresponding to the matrix ``Q`` in
``[x]_d^TQ[x]_d`` for a given order ``n``.
Example:
For a third-order SOS polynomial, ``[x]_3 = [1, x, x^2, x^3]`` and the coefficient matrix ``Q`` is
```math
Q = \\begin{bmatrix}
Q_{11} & Q_{12} & Q_{13} & Q_{14} \\\\
Q_{21} & Q_{22} & Q_{23} & Q_{24} \\\\
Q_{31} & Q_{32} & Q_{33} & Q_{34} \\\\
Q_{41} & Q_{42} & Q_{43} & Q_{44}
\\end{bmatrix} \\\\
```
If ``n=3`` we have
```julia
julia> get_inds_2d(SOS(3), 3)
CartesianIndex{2}[CartesianIndex(2, 2), CartesianIndex(3, 1)]
```
since ``Q_{22}`` and ``Q_{31}`` correspond to third order terms of ``[x]_d^TQ[x]_d``.
"""
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
    inds = map(x -> CartesianIndex(x[1], x[2]), inds)
    return inds
end

export get_inds_2d

"""
    get_inds(sos, n, dim=1)

Returns flattened indices of get_inds_2d, for indexing a lower triangular matrix with a vector. \\
If dim == 1, return the vectorized indices. \\
If dim > 1, each entry in the lower triangular matrix is being indexed on a separate row, useful for constructing a matrix of constraints where each entry of the coefficient matrix, ``Q`` is set separately. \\
It is the user/solver responsibility to eliminate rows of zeros!
"""
function get_inds(sos::SOS, n, dim=1)
    inds = get_inds_2d(sos, n)
    inds = lower_triangular_from_2d_idx(sos.n, inds)
    if dim == 1
        inds = map(x -> CartesianIndex(1, x), inds)
    else
        inds = map(x -> CartesianIndex(x, x), inds)
    end
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

function evaluate_monomials(p, v)
    ps = collect(monomials(p))
    vs = map(kv -> kv[2](v), reduce(vcat, enumerate(ps)))
    return vs
end

function variable_from_term(term)
    return effective_variables(monomial(term))[end]
end

function polynomial_substitute(p, s)
    p_subs = subs(p, s)
    p_result = p
    if isconstant(p_subs)
        return p_subs
    end
    for term in terms(p_subs)
        c = coefficient(term)
        v = variable_from_term(term)
        p_result = subs(p_result, v=>c)
    end
    return p_result
end

export polynomial_substitute

"""
    add_polynomial_equality_constraint(sos, b, p)

Set an equality constraint on the polynomial to equal the value b.
"""
function add_polynomial_equality_constraint(sos::SOS, b, p)
    A = zeros((1, get_size(sos.cones[end])))
    c = coefficient(p, constant_monomial(p))
    b -= c
    for term in terms(p - c)
        term_var = monomial(term)
        term_idx = sos.char_map[term_var][end]
        idx = term_idx[end]
        A[idx] = coefficient(term)
    end
    return A, [b]
end

"""
    get_vec_inds_map(A)

Given a list of monomials from A, returns a dictionary from monomial to list of indices to the vectorized coefficient matrix, Q.
"""
function get_vec_inds_map(A)
    T = Tuple{Tuple{Int64, Int64}, Int64}
    d = DefaultDict{DynamicPolynomials.Monomial, Vector{T}}(Vector{T})
    N = size(A, 1)
    i = 1
    for n in 1:N
        for m in n:N
            push!(d[A[m, n]], ((m, n), i))
            i += 1
        end
    end
    return d
end

"""
    get_polynomial_equality_constraint_from_coefficients(p, rhs)

Returns the affine constraints as a matrix required for determining if a polynomial, p is SOS.
"""
function get_polynomial_equality_constraint_from_coefficients(sos_p, rhs=0)
    p = sos_p.p - rhs
    ps = collect(monomials(p))
    char_map = sos_p.sos.char_map
    coeffs = DynamicPolynomials.coefficients(p)
    m = length(coeffs)
    n = Int(sos_p.sos.n * (sos_p.sos.n + 1) / 2)
    A = zeros((m, n))
    # assemble affine constraints by matching coefficients (page 62 of Blekherman, Parrilo, Thomas)
    for (i, m) in enumerate(ps)
        v = char_map[m]
        for j in v
            k, l = j[begin]
            if k == l # coefficient on the diagonal
                A[i, j[end]] = 1
            else # coefficient off diagonal
                A[i, j[end]] = 2 # two cross-terms e.g. ab + ba
            end
        end
    end
    b = coeffs
    return A, b, sos_p.sos.n, ps
end

export get_polynomial_equality_constraint_from_coefficients

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

function add_polynomial_inequality_constraint(sos::SOS, b, p, v)
    p = evaluate_monomials(p, v)
    add_polynomial_inequality_constraint(sos, b, p)
end

"""
    sos_to_qp(sos)

Get the Cone QP object representing the SOS optimization problem
"""
function sos_to_qp(sos::SOS)
    num_vars = size(sos.A)[2]
    G = -Matrix(1.0I, num_vars, num_vars)
    h = zeros(size(sos.G)[1])
    cone_qp = ConeQP(sos.A, G, sos.P, sos.b, sos.c, h, sos.cones)
    return cone_qp
end

export SOS
export get_inds
export set_objective
export set_diagonal_Q_constraint
export add_polynomial_equality_constraint
export add_polynomial_inequality_constraint
export sos_to_qp