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

function decompose(f)
    deg = DynamicPolynomials.maxdegree(f)
	vars = DynamicPolynomials.variables(f)
	basis = DynamicPolynomials.monomials(vars, 0:deg) # basis_constraints
	basis_half = DynamicPolynomials.monomials(vars, 0:deg÷2) # basis_psd

	perm_G = SymbolicWedderburn.PermGroup([perm"(1,2)", Perm([2:n; 1])])
	wedderburn = WedderburnDecomposition(
		Float64,
		perm_G,
		VariablePermutation(x),
		basis,
		basis_half,
		semisimple=false)

	M = let basis_constraints = StarAlgebras.basis(wedderburn)
		[basis_constraints[x*y] for x in basis_half, y in basis_half]
	end
	M_orb = similar(M, eltype(wedderburn))

    C = DynamicPolynomials.coefficients(f, SymbolicWedderburn.basis(wedderburn))
	psds = SymbolicWedderburn.direct_summands(wedderburn)

    ivs = SymbolicWedderburn.invariant_vectors(wedderburn)
    b = zeros(length(ivs))
    for (i, iv) in enumerate(ivs)
        c = dot(C, iv)
	    M_orb_ivc = invariant_constraint!(M_orb, M, iv)
        Mπs = SymbolicWedderburn.diagonalize(M_orb_ivc, wedderburn)
        # sum(dot(Mπ, Pπ) for (Mπ, Pπ) in zip(Mπs, psds) if !iszero(Mπ)) == c
        if !iszero(Mπ)
            b[i] = c
        else
            b[i] = 0
        end
    end
    return Mπs, b
end

function set_objective(c)
    c[end] = -1
    return c
end

# ------------------------------------------------------------------------
# NOTES
# The PSD constraint matrix "G" is a diagonal matrix
# The vector "h" is the zero vector

# The matrix "A" has the following structure
# 1 ... 1 0 ... 0 0 ... 0
# 0 ... 0 1 ... 1 0 ... 0
# 0 ... 0 0 ... 0 1 ... 1
# The vector "b" is just dot(C, iv)

# The vector "c" is [0 ... 0 t=1]
# The objective is to maximize c*x, i.e. minimize -c*x
# ------------------------------------------------------------------------
function sos_to_qp(f)
    Mπs, b = decompose(f)
    num_vars = Int(sum([size(x, 1) * (size(x, 1) + 1)/2 for x in Mπs]))
    A = zeros((length(b), num_vars + 1))
    j = 1
    for i in axes(A, 1)
        idx = j
        j += size(Mπs[i], 1)
        A[i, idx:j] .= 1
    end
    c = zeros(num_vars + 1)
    c = set_objective(c)
    G = Matrix{Float64}(I, num_vars + 1, num_vars + 1)
    h = zeros(num_vars + 1)
    P = zeros((num_vars + 1, num_vars + 1))

    # construct problem
    cones::Vector{Cone} = []
    for x in Mπs
        p = size(x, 1)
        push!(cones, PSDCone(p))
    end
    push!(cones, NonNegativeOrthant(1))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end