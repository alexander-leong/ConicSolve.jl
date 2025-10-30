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

    for iv in SymbolicWedderburn.invariant_vectors(wedderburn)
        c = dot(C, iv)
	    M_orb_ivc = invariant_constraint!(M_orb, M, iv)
        Mπs = SymbolicWedderburn.diagonalize(M_orb_ivc, wedderburn)
        # sum(dot(Mπ, Pπ) for (Mπ, Pπ) in zip(Mπs, psds) if !iszero(Mπ)) == c
    end
end

# function mat_from_row_blocks(gs)
#     row_dims = [size(x)[1] for x in gs]
#     col_dims = [size(x)[2] for x in gs]
#     prepend!(row_dims, 1)
#     prepend!(col_dims, 1)
#     num_rows = sum(row_dims)
#     num_cols = maximum(col_dims)
#     G = zeros(num_rows, num_cols)
#     row_idx = 1
#     for (i, g) in enumerate(gs)
#         row_idx += row_dims[i+1]
#         row_idx_n += row_dims[i+2]
#         col_idx += col_dims[i+1]
#         col_idx_n += col_dims[i+2]
#         rows = row_idx:row_idx_n
#         cols = col_idx:col_idx_n
#         G[rows, cols] = g
#     end
#     return G
# end

function set_objective()
end

function sos_to_qp(f)
    decompose(f)
    # G = mat_from_row_blocks()

    # construct problem
end