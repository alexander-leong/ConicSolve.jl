#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using AbstractPermutations
using DynamicPolynomials
using GroupsCore
using LinearAlgebra
using PermutationGroups
using StarAlgebras
using SparseArrays
using SymbolicWedderburn

mutable struct SOS_Symmetric_Group
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

function get_lt_idx(n)
    i = 1
    inds = []
    for l in 1:n
        for k in i:n
            push!(inds, CartesianIndex((k, l)))
        end
        i += 1
    end
    return inds
end

function get_lt_vals(A)
    n = size(A, 1)
    inds = get_lt_idx(n)
    lt_vals = map(i -> A[i], inds)
    return lt_vals
end

U_map = Dict()

function reduce_psd(A, i)
    A, U, S = tsvd(A)
    U_map[i] = (U, S)
    return diagm(S)
end

function set_A_i!(A, Mπs)
	num_vars = size(A, 2)
    j = 0
    m = 0
	# println(length(axes(Mπs, 1)))
    for Mπs_j in Mπs
        idx = j
		idx += 1
        j += Int(size(Mπs_j, 1) * (size(Mπs_j, 1) + 1) / 2)
        # get vectorized lower triangular entries of Mπs[i]
        Mπs_j, U, S = tsvd(Mπs_j)
        # eigmax used for computing regularized term cannot handle complex eigenvalues
        Mπs_j = regularize(Mπs_j)
        # cn = cond(Mπs_j)
        # if cn > 1e2
        S = svd(Mπs_j).S
        σ_max, σ_min = S[1], S[end]
        # println("WARNING: Poorly conditioned PSD condition generated! (condition number = $(cn))")
        println("PSD condition generated (n, σ_max, σ_min = $(length(S)), $(σ_max), $(σ_min))")
        if length(S) == 3 && σ_max > 0 && isapprox(σ_min, 0)
            println(Mπs_j)
        end
        # end
		Mπ_vec = get_lt_vals(Mπs_j)
		
        if iszero(Mπ_vec)
            continue
        end
        a = zeros((1, num_vars))
        a[idx:j] .= Mπ_vec
        A = vcat(A, a)
        m += 1
    end
    return A, m
end

# rank revealing qr, remove redundant constraints
function rrqr(A, b, tol=1e-3)
    F = qr(A', ColumnNorm())
    # get list of row indices where diagonal elements of R satisfy tol
    inds = [i for (i, v) in enumerate(diag(F.R)) if abs(v) >= tol]
    inds = [x[2] for x in findall(val -> val == 1, F.P'[inds, :])]
    # reduced A, b
    rA = A[inds, :]
    rb = b[inds]
    return rA, rb
end

function tsvd(A, tol=1e-9)
    F = svd(A)
    S::Vector{Float64} = []
    for v in F.S
        if v >= tol
            push!(S, v)
        end
    end
    U = F.U[:, 1:length(S)]
    Vt = F.Vt[1:length(S), :]
    A = U * diagm(S) * Vt
    return Hermitian(A), U, S
end

function regularize(A, tol=1e-9)
    eps = tol * eigmax(A)
    A = A + eps * I
    return A
end

function decompose(f, n, x, num_additional_vars=0)
    deg = DynamicPolynomials.maxdegree(f)
	vars = DynamicPolynomials.variables(f)
	basis = DynamicPolynomials.monomials(vars, 0:deg) # basis_constraints
    basis_half = DynamicPolynomials.monomials(vars, 0:deg÷2) # basis_psd

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
    # Matrix A defining equality constraints in Ax = b
    total_num_vars = num_vars + num_additional_vars
    A = Matrix{Float64}(undef, 0, Int(total_num_vars))
    # Vector b defining equality constraints in Ax = b
    b = []

    basis_constraints = SymbolicWedderburn.basis(wedderburn)
    C = DynamicPolynomials.coefficients(f, basis_constraints)
    
    ivs = SymbolicWedderburn.invariant_vectors(wedderburn)
    for iv in ivs
        c = dot(C, iv)
	    M_orb_ivc = invariant_constraint!(M_orb, M, iv)
        Mπs = SymbolicWedderburn.diagonalize(M_orb_ivc, wedderburn)
        A, m = set_A_i!(A, Mπs)
        b = append!(b, fill(c, m))
    end

    A, b = rrqr(A, b)
    cn = cond(A)
    if cn > 1e2
        println("WARNING: Poorly conditioned equality matrix defined! (condition number = $(cn)")
    end
    return A, b, num_vars, psds
end

function get_value_in_original_basis(psds, x)
    # get block diagonal form from solution vector x
    n_i = [size(X, 1) for X in psds_mat]
    inds = [1, cumsum(n_i)...]
    X_i = [mat(x[inds[i]:inds[i+1]]) for (i, _) in enumerate(psds)]

    # apply change of basis on bilinear form (M = psds * blockdiag(x) * psds^T)
    Ms = []
    for (i, psd) in enumerate(psds)
        M_i = vec(psd * X_i[i] * psd')
        push!(Ms, M_i)
    end

    # convert to vectorized form
    M = vcat(Ms...)
    return M
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

export decompose