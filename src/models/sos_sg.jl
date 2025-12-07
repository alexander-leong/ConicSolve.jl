#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using AbstractPermutations
using DynamicPolynomials
using GroupsCore
using LinearAlgebra
using MultivariateBases
using PermutationGroups
using StarAlgebras
using SparseArrays
using SymbolicWedderburn

include("../arrayutils.jl")

mutable struct SOS_Symmetric_Group
    basis
    basis_half
    basis_type
    deg
    equality_constraint_indices
    f
    SOS_Symmetric_Group() = new()
    SOS_Symmetric_Group(basis,
                        basis_half,
                        basis_type,
                        deg,
                        equality_constraint_indices,
                        f) = new(basis,
                            basis_half,
                            basis_type,
                            deg,
                            equality_constraint_indices,
                            f)
end

function SymbolicWedderburn.action(a::SymbolicWedderburn.Action, el::GroupElement, poly::DynamicPolynomials.AbstractPolynomial)
    return sum(SymbolicWedderburn.action(a, el, term) for term in DynamicPolynomials.terms(poly))
end

function SymbolicWedderburn.action(a::SymbolicWedderburn.Action, el::GroupElement, term::DynamicPolynomials.AbstractTerm)
    return DynamicPolynomials.coefficient(term) * SymbolicWedderburn.action(a, el, DynamicPolynomials.monomial(term))
end

struct VariablePermutation{V} <: SymbolicWedderburn.ByPermutations
    variables::V
end

function SymbolicWedderburn.action(
    a::VariablePermutation,
    g::Permutation,
    m::AbstractMonomial
)
    v = a.variables
    return m(v => SymbolicWedderburn.action(a, g, v))
end

function invariant_constraint!(
    M_orb::AbstractMatrix{<:AbstractFloat},
    M::Matrix{<:Integer},
    invariant_vec::SparseVector
)
    M_orb .= zero(eltype(M_orb))
    for i in eachindex(M)
        if M[i] ∈ SparseArrays.nonzeroinds(invariant_vec)
            M_orb[i] += invariant_vec[M[i]]
        end
    end
    return M_orb
end

function set_A_i!(A, Mπs, equality_constraint_indices)
    j = 0
    m = 0
    for Mπs_j in Mπs
        idx = j
		idx += 1
        j += Int(size(Mπs_j, 1) * (size(Mπs_j, 1) + 1) / 2)
		
        # Omit generated redundant PSD constraint
        if iszero(Mπs_j)
            continue
        end
        # get vectorized lower triangular entries of Mπs[i]
        Mπ_vec = get_lt_vals(Mπs_j)

        a = get_constraint(A, Mπ_vec, idx)
        A = vcat(A, a)
        push!(equality_constraint_indices, (idx, j))
        m += 1
    end
    return A, m
end

function get_num_vars(Mπs_j, j)
    j = Int(size(Mπs_j, 1) * (size(Mπs_j, 1) + 1) / 2)
    return j
end

function get_constraint(A, Mπ_vec, idx)
    num_vars = size(A, 2)
    a = zeros((1, num_vars))
    j = idx + length(Mπ_vec) - 1
    # set values
    a[idx:j] .= Mπ_vec
    return a
end

# rank revealing qr, remove redundant constraints
function rrqr(A, b, tol=1e-9)
    F = qr(A', ColumnNorm())
    # get list of row indices where diagonal elements of R satisfy tol
    inds = [i for (i, v) in enumerate(diag(F.R)) if abs(v) >= tol]
    inds = [x[2] for x in findall(val -> val == 1, F.P'[inds, :])]
    # reduced A, b
    rA = A[inds, :]
    rb = b[inds]
    return rA, rb, inds
end

function get_monomial_basis(f, deg)
	vars = DynamicPolynomials.variables(f)
	basis = DynamicPolynomials.monomials(vars, 0:deg) # basis_constraints
    basis_half = DynamicPolynomials.monomials(vars, 0:deg÷2) # basis_psd
    return basis, basis_half
end

# TODO: remove unsupported logic
# NOTE: SymbolicWedderburn.jl does not currently support other basis types
function get_chebyshev_basis(f, deg)
    basis, basis_half = get_monomial_basis(f, deg)
    basis = basis_covering_monomials(ChebyshevBasis, basis)
    basis_half = basis_covering_monomials(ChebyshevBasis, basis_half)
    return basis.polynomials, basis_half.polynomials
end

function wedderburn_decompose(f, n, x, num_additional_vars=0)
    deg = DynamicPolynomials.maxdegree(f)
    println("Polynomial function f has degree: $(deg)")
    println("Symmetric Group $(n)")
    basis, basis_half = get_monomial_basis(f, deg)

	pg = PermGroup([perm"(1,2)", Perm([2:n; 1])])
	wedderburn = WedderburnDecomposition(
		Float64,
		pg,
		VariablePermutation(x),
		basis,
		basis_half,
		semisimple=false)

	M = let basis_constraints = StarAlgebras.basis(wedderburn)
		[basis_constraints[x*y] for x in basis_half, y in basis_half]
	end
	M_orb = similar(M, eltype(wedderburn))

    psds = SymbolicWedderburn.direct_summands(wedderburn)
    num_vars = Int(sum([size(psd, 1) * (size(psd, 1) + 1) / 2 for psd in psds]))

    # FIXME: relocate this logic outside
    U_map = Dict()
    # Matrix A defining equality constraints in Ax = b
    total_num_vars = num_vars + num_additional_vars
    A = Matrix{Float64}(undef, 0, Int(total_num_vars))
    # Vector b defining equality constraints in Ax = b
    b = []

    basis_constraints = SymbolicWedderburn.basis(wedderburn)
    C = DynamicPolynomials.coefficients(f, basis_constraints)
    
    ivs = SymbolicWedderburn.invariant_vectors(wedderburn)
    equality_constraint_indices = []
    for iv in ivs
        c = dot(C, iv)
	    M_orb_ivc = invariant_constraint!(M_orb, M, iv)
        Mπs = SymbolicWedderburn.diagonalize(M_orb_ivc, wedderburn)
        A, m = set_A_i!(A, Mπs, equality_constraint_indices)
        b = append!(b, fill(c, m))
        # println("End of Invariant Vector")
    end
    println("Length of A: $(size(A, 1))")
    println("Length of Equality Inds: $(length(equality_constraint_indices))")

    A, b, inds = rrqr(A, b)
    equality_constraint_indices = equality_constraint_indices[inds]
    cn = cond(A)
    if cn > 1e2
        println("WARNING: Poorly conditioned equality matrix defined! (condition number = $(cn))")
    end

    sos_symmetric_group = SOS_Symmetric_Group(
        basis,
        basis_half,
        "monomial",
        deg,
        equality_constraint_indices,
        f
    )
    println("Num rows in A: $(size(A, 1))")
    println("End of decompose, returning...")
    return A, b, num_vars, psds, sos_symmetric_group
end

function get_mat_vec_len(psds)
    # number of elements in each lower triangular psd matrix
    n_i = [get_lt_vec_len(X) for X in psds]
    return n_i
end

function get_value_in_original_basis(summands, x)
    # TODO useless for now
    # performs change of basis for an endomorphism
    # see pg.118 of MO13-Blekherman-Parrilo-Thomas
    
    # get block diagonal form from solution vector x
    n_i = get_mat_vec_len(summands)
    inds = [0, cumsum(n_i)...]
    xs = []
    for i in 1:length(inds)-1
        push!(xs, x[inds[i]+1:inds[i+1]])
    end
    X = block_diagonalize_from_lt_vecs(xs)
    println("Dimensions of X: $(size(X))")

    # construct change of basis matrix T
    Q = qr(vcat(summands...)).Q
    m = size(summands[1], 2)
    n = size(Q, 2)
    T = Matrix{Float64}(I, m, m)
    println("Change of basis matrix dims: $(m)")
    println("Dimensions of alternate Wedderburn basis: $(n)")
    T[1:n, 1:n] = collect(Q)

    # apply change of basis on bilinear form (M = psds * blockdiag(x) * psds^T)
    X = T * X * T'
    println("YES")
    return X
end

# ------------------------------------------------------------------------
# NOTES
# The PSD constraint matrix "G" is a diagonal matrix
# The vector "h" is the zero vector

# The constraint dot(Mπ, Pπ) for (Mπ, Pπ) in zip(Mπs, psds) if !iszero(Mπ) == c
# is implemented by the matrix "A" having the following structure:
# vec(Mπ_1)  0 ... 0   0 ... 0
#  0 ... 0  vec(Mπ_2)  0 ... 0
#  0 ... 0   0 ... 0  vec(Mπ_n)
# The vector "b" is just dot(C, iv)
# ------------------------------------------------------------------------
function sos_to_qp(f, n, x)
    A, Mπs, b = decompose(f, n, x)
    num_vars = Int(sum([size(x, 1) * (size(x, 1) + 1)/2 for x in Mπs]))
    c = zeros(num_vars + 2)
    # The vector "c" is [0 ... 0 -1]
    c = set_objective(c)
    G = Matrix{Float64}(I, num_vars + 2, num_vars + 2)
    h = zeros(num_vars + 2)
    P = zeros((num_vars + 2, num_vars + 2))

    # construct problem
    cones::Vector{Cone} = []
    for x in Mπs
        p = size(x, 1)
        push!(cones, PSDCone(p))
    end
    push!(cones, NonNegativeOrthant(2))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end

export SOS_Symmetric_Group
export wedderburn_decompose
export get_constraint
export get_mat_dim
export get_mat_from_lt_vec
export get_value_in_original_basis
export get_vec_from_lt_mat