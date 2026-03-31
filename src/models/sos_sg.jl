#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using AbstractPermutations
using ConicSolve
using DynamicPolynomials
using GroupsCore
using LinearAlgebra
using MultivariateBases
using PermutationGroups
using StarAlgebras
using SparseArrays
using SymbolicWedderburn

mutable struct SOS_Symmetric_Group
    basis
    basis_half
    basis_type
    deg
    f
    SOS_Symmetric_Group() = new()
    SOS_Symmetric_Group(basis,
                        basis_half,
                        basis_type,
                        deg,
                        f) = new(basis,
                            basis_half,
                            basis_type,
                            deg,
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

function set_A_i!(program, Mπs, b, equality_constraint_indices)
    j = 0
    m = 0
    for Mπs_j in Mπs
        idx = j
		idx += 1
        j += Int(size(Mπs_j, 1) * (size(Mπs_j, 1) + 1) / 2)
        
        if (idx, j) in keys(equality_constraint_indices)
            cone = equality_constraint_indices[(idx, j)]
        else
            p = size(Mπs_j, 1)
            cone = add_variable(program, PSDCone(), p)
            equality_constraint_indices[(idx, j)] = cone
        end
		
        # Omit generated redundant PSD constraint
        if iszero(Mπs_j)
            continue
        end
        
        # get vectorized lower triangular entries of Mπs[i]
        Mπ_vec = get_lt_vals(Mπs_j)

        add_affine_constraint(program, cone, Mπ_vec, b)
        m += 1
    end
    return m
end

function get_num_vars(Mπs_j, j)
    j = Int(size(Mπs_j, 1) * (size(Mπs_j, 1) + 1) / 2)
    return j
end

function get_constraint(A::AbstractArray{Float64}, Mπ_vec::AbstractArray{Float64}, idx::Int32)
    num_vars = size(A, 2)
    a = zeros((1, num_vars))
    j = idx + length(Mπ_vec) - 1
    # set values
    a[idx:j] .= Mπ_vec
    return a
end

function get_monomial_basis(f, deg)
	vars = DynamicPolynomials.variables(f)
	basis = DynamicPolynomials.monomials(vars, 0:deg) # basis_constraints
    basis_half = DynamicPolynomials.monomials(vars, 0:deg÷2) # basis_psd
    return basis, basis_half
end

"""
    get_subproblem(in_program, cone)

Constructs a program with objective and constraints with respect to the given cone.
"""
function get_subproblem(in_program::ConicSolve.ConeQP,
                        cone::Cone)
    # given initial problem, in_program, construct subproblem from constraints and objective with respect to given cone
    out_program = ConicSolve.ConeQP()

    affine_constraints = find_affine_constraints_by_cone(in_program, cone)
    for constraint in affine_constraints
        add_affine_constraint(out_program, cone, constraint.lhs, constraint.rhs)
    end
    inequality_constraints = find_inequality_constraints_by_cone(in_program, cone)
    for constraint in inequality_constraints
        add_inequality_constraint(out_program, cone, constraint.lhs, constraint.rhs)
    end

    inds = get_indices_of_constraint(in_program, cone)
    obj = in_program.c[inds]
    set_objective(out_program, cone, obj)

    add_variable(out_program, cone, cone.p)

    out_program = build_program(out_program, true)
    return out_program
end

export get_subproblem

"""
    wedderburn_decompose(program, f, n, x, num_additional_vars=0)

Performs a Wedderburn Decomposition on the polynomial function f with respect to Symmetric Group n and variable x.

Example
```julia
n = 4

f =
    1 +
    sum(x .+ 1) +
    sum((x .+ 1) .^ 2)^4 +
    sum((x .+ x') .^ 2)^2 * sum((x .+ 1) .^ 2)

program = ConeQP()
summands, sos_symmetric_group = wedderburn_decompose(program, f, n, x)
```
Returns
- the summands object for the change of basis between the monomial basis and fixed-point subspace.
- the SOS symmetric group object corresponding to the wedderburn decomposition
"""
function wedderburn_decompose(program::ConeQP,
                              f::DynamicPolynomials.Polynomial,
                              n::Int,
                              x,
                              num_additional_vars=0)
    deg = DynamicPolynomials.maxdegree(f)
    @info "### Executing Symmetry Reduction ###"
    @info "Polynomial function f has degree: $(deg)"
    @info "Symmetric Group $(n)"
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

    summands = SymbolicWedderburn.direct_summands(wedderburn)
    num_vars = Int(sum([size(psd, 1) * (size(psd, 1) + 1) / 2 for psd in summands]))

    total_num_vars = num_vars + num_additional_vars
    program.A = Matrix{Float64}(undef, 0, Int(total_num_vars))

    basis_constraints = SymbolicWedderburn.basis(wedderburn)
    C = DynamicPolynomials.coefficients(f, basis_constraints)
    
    ivs = SymbolicWedderburn.invariant_vectors(wedderburn)
    equality_constraint_indices = Dict()
    for iv in ivs
        c = dot(C, iv)
	    M_orb_ivc = invariant_constraint!(M_orb, M, iv)
        Mπs = SymbolicWedderburn.diagonalize(M_orb_ivc, wedderburn)
        set_A_i!(program, Mπs, c, equality_constraint_indices)
    end

    sos_symmetric_group = SOS_Symmetric_Group(
        basis,
        basis_half,
        "monomial",
        deg,
        f
    )
    
    @info "End of decompose, returning..."
    return summands, sos_symmetric_group
end

function get_mat_vec_len(psds)
    # number of elements in each lower triangular psd matrix
    n_i = [get_lt_vec_len(X) for X in psds]
    return n_i
end

"""
    get_reduced_solution(summands, xs)

# Parameters:
* `summands`: The DirectSummands data structure returned by wedderburn_decompose
* `xs`: The vector of solution vectors to each subproblem

Gets the solution in terms of the original basis.
"""
function get_reduced_solution(summands, xs)
    n = size(summands[1], 2)
    result = zeros((n, n))
    for (summand, x) in zip(summands, xs)
        result += summand' * mat(x) * summand
    end
    return result
end

export SOS_Symmetric_Group
export wedderburn_decompose
export get_constraint
export get_mat_dim
export get_mat_from_lt_vec
export get_reduced_solution
export get_vec_from_lt_mat