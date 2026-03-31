# Reduction Methods

## Introduction

Algebraic methods such as symmetry reduction and face reduction can be used to complement numerical methods by exploiting problem structure to reduce problem size and improve numerical conditioning that causes solvers to stall and fail. We discuss these methods below.

## Symmetry Reduction

Symmetry reduction plays a crucial role in reducing the size of the problem and improving numerical conditioning by finding a special projection map that takes the problem and maps it onto itself (the fixed point subspace). 

Applying symmetry reduction before face reduction is what enables massive computational efficiency gains by simultaneously performing dimensional reduction and improving numerical conditioning.

The size of an SDP problem (measured by the number of variables) dominates the overall solve time. For a fourth-order polynomial (as in our sample problem) without symmetry reduction, we would have had 70 monomial terms or $70*71/2$ variables to solve for. SymbolicWedderburn.jl is a package used by ConicSolve.jl to obtain a direct sum decomposition of the system to solve for. This results in a block diagonalization of the problem into 4 smaller SDP problems, the largest SDP $13*14/2$ variables to solve for.

The Wedderburn decomposition is used to give a direct-sum representation of the algebra defined by the group action over Symmetric Group $d$.

This is expressed as
``V = \oplus_{i=1}^rV_i``
characterized by the set of projections
```math
\pi_i : V \mapsto V_i
```

If we denote the jth diagonal block of ``V_i`` as ``V_{ij}`` we have the affine constraints for the ith SDP given by

```math
A^i = \Sigma_{j} V_i^j \qquad ; \qquad
b^i = C \cdot V_i^j
```

where ``C`` are the coefficients of the degree ``2d`` polynomial ``p(x)``.

The set of projections ``\pi_i`` is chosen so that a block-diagonalized decomposition gives separate SDP programs where the affine constraints given by ``A`` and ``b`` is just the summation over the respective diagonalized blocks.

Reconstructing the solution from the fixed point subspace to obtain the coefficients of the 70 monomial terms is just the sum over each of the direct summand elements and the solution ``x`` (in matrix form, ``X``).

```math
\Sigma_{i=1}^n V_iXV_i^T
```

```julia
n = 4

f =
    1 +
    sum(x .+ 1) +
    sum((x .+ 1) .^ 2)^4 +
    sum((x .+ x') .^ 2)^2 * sum((x .+ 1) .^ 2)

program = ConeQP()
summands, sos_symmetric_group = wedderburn_decompose(program, f, n, x)

vars = program.vars
for cone in vars.cones
    add_default_inequality_constraint(program, cone)
    n = ConicSolve.get_size(cone)
    c = ones(n)
    set_objective(program, cone, c)
end

program = build_program(program)
```

We can get the solution from the summands and solution vector like follows
```julia
solution = get_reduced_solution(summands, x_vec)
```

## Face Reduction

Symmetry reduction does not remove the structural degeneracy inherent in quantum control problems, which causes stalling and numerical failure of interior point method-based solvers. We apply face reduction as a numerical method to remove structural degeneracy where the solution to the problem lies on a lower dimensional subspace. The following algorithm implements Algorithm 1.1 as described by Permenter 2017.

```math
\begin{array}{l}
\textbf{Algorithm 1} \text{ Face Reduction as SDP}(A) \\
\hspace{1em} \textbf{Inputs: } \text{affine set }A \\
\hspace{1em} \textbf{Output: } \text{a face }F \\
\hspace{1em} \textbf{begin} \\
\hspace{2em} \textbf{while } \text{(*) is feasible } \textbf{do} \\
\hspace{3em} \textbf{Expose face by solving the following SDP} (*) \\
\hspace{4em}
\begin{aligned}\text{minimize}\qquad & 0 \\
\text{subject to}\qquad & Ax = b \\
\qquad & x \in F \end{aligned} \\
\hspace{3em} \textbf{Update face} \\
\hspace{4em} \tilde{X} = U_r\Lambda_rU_r^T \\
\hspace{2em} \textbf{end while} \\
\hspace{2em} \textbf{Return } F \\
\hspace{1em} \textbf{end}
\end{array}
```

The expression $U_r\Lambda_rU_r^T$ is found by taking the truncated singular value decomposition of $X$ where $r$ is determined by a numerical threshold $\eta_\lambda$.

The performance of face reduction depends on:
- the solver's ability to get a near zero duality gap
- a numerical threshold on $S$ that is as small as practical

The following optimization problem now solves for the solution in terms of the minimal face

```math
\begin{equation} \label{eq:9} \begin{aligned}\text{minimize}\qquad & \langle\text{vec}(U^T\text{mat}(c)U), x\rangle \\
\text{subject to}\qquad & A = \lbrace x \in V : Ax = b \rbrace \end{aligned}\end{equation}
```

The solution is then mapped back to the original subspace by reversing the projection operations
$X_n = U^TX_{n-1}U$ which is
``\begin{equation} \label{eq:10} X_{n-1} = UX_kU^T\end{equation}``

```julia
ϵ = 1e-6 # absolute tolerance
η_eps = 2e-2 # absolute tolerance to determine exposed face (using duality gap)
η_lambda = 1e-3 # absolute tolerance to remove near redundant constraints
p = 2 # order of Chebyshev expansion for approximating matrix exponential
x_vec, _ = run_fr_solver(solver, true, η_eps, η_lambda)
```