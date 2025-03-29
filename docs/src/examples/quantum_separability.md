# Quantum Separability

## Introduction

Detecting whether the Quantum state of a Quantum system is separable or not is a fundamental problem in Quantum information theory and applications.

In this example we'll look at a bipartite quantum state in a 2x2 Hilbert space. The same concepts apply when developing SDP formulations for 3x3 systems and so forth. Additional conditions for determining separability for the 3x3 case and beyond (referred to as a hierarchy) do apply. We proceed with an application of the DPS [Doherty, Parrilo, Spedalieri hierarchy](https://arxiv.org/pdf/quant-ph/0308032) as follows.

## How to run the example

1. Ensure that you have the necessary dependencies installed. These can be installed by executing the following commands:
```julia
julia> ]
pkg> activate .
pkg> add ConicSolve
julia> exit()
```

### Explanation

The bases for a 2-dimensional Hilbert space such that they satisfy:
```math
Tr[\sigma_i^X\sigma_j^X] = \alpha_X\delta_{ij} \hspace{1cm} and \hspace{1cm} Tr[\sigma_i^X] = \delta_{i1}
``` for subsystem ``X`` is given by the Pauli matrices:
```math
\sigma_0 = \begin{bmatrix}
1/\alpha_X & 0 \\
0 & 1/\alpha_X \\
\end{bmatrix}
\hspace{1cm}
\sigma_1 = \begin{bmatrix}
0 & 1/\alpha_X \\
1/\alpha_X & 0 \\
\end{bmatrix}
\hspace{1cm}
\sigma_2 = \begin{bmatrix}
0 & -i/\alpha_X \\
i/\alpha_X & 0 \\
\end{bmatrix}
\hspace{1cm}
\sigma_3 = \begin{bmatrix}
1/\alpha_X & 0 \\
0 & -1/\alpha_X \\
\end{bmatrix}
```
where ``\alpha_X = 2``.

The PPT extension ``\tilde{\rho}`` can be written as a set of Linear Matrix Inequality (LMI) for an SDP.
```math
G_0 = \sum_{j} \rho_{1j}\sigma_1^A \otimes \sigma_j^B \otimes \sigma_1^A + \sum_{i=2, j=1} \rho_{ij}\{\sigma_i^A \otimes \sigma_j^B \otimes \sigma_1^A + \sigma_1^A \otimes \sigma_j^B \otimes \sigma_i^A\}
```
```math
G_{iji} = \sigma_i^A \otimes \sigma_j^B \otimes \sigma_i^A \hspace{1cm} i \ge 2
```
```math
G_{ijk} = (\sigma_i^A \otimes \sigma_j^B \otimes \sigma_k^A + \sigma_k^A \otimes \sigma_j^B \otimes \sigma_i^A) \hspace{1cm} k \gt i \ge 2
```
where ``\rho_{ij}`` is defined as
```math
\rho_{ij} = \alpha_A^{-1}\alpha_B^{-1}Tr[\rho\sigma_i^A \otimes \sigma_j^B]
```

So in Conic form the ``G`` matrix can be written as
```math
G = \begin{bmatrix}
svec(G_0) & svec(G_{iji}) & svec(G_{ijk})
\end{bmatrix}
```
The ``svec(G_{iji})`` and ``svec(G_{ijk})`` are hcat (horizontal concatenate) together with ``svec(G_0)``

We must not forget the partial transposes of ``\tilde{\rho}`` with respect to subsystem A, ``\tilde{\rho}^{T_A}`` and subsystem B, ``\tilde{\rho}^{T_B}``. So once these are computed the final ``G`` matrix which we call ``F`` for lack of better notation can be written as:
```math
F = \begin{bmatrix}
G \\
G^{T_A} \\
G^{T_B} \end{bmatrix}
```

We now take F (as the variable G in code) to enforce the PSD condition for the SDP feasibility problem:
```math
\begin{aligned}
\text{minimize}\qquad & 0 \\
\text{subject to}\qquad &
F \succeq 0 \\
\end{aligned}
```

#### Data Acquisition
This is a simple toy problem setup. No data has been imported in this example.

#### Solve the problem

(i) We create a ConeQP object that represents the SDP problem to solve
`cone_qp = get_qp()`.

(ii) We pass the ConeQP object to the solver `solver = Solver(cone_qp)`.

(iii) Then when we're ready we call optimize! passing the solver object `optimize!(solver)`.

(iv) We can access the solution by accessing the primal solution from the solver `x = get_solution(solver)`.

#### Get the solution
For the given density matrix ``\rho`` we see all residuals tend to zero suggesting that the quantum state is separable.

#### Further Comments
The DPS hierarchy is a complete set of criteria for determining separability of quantum states in a bipartite quantum system. The PSD condition is only one of the tests (not necessarily sufficient) and the reader should consult the DPS paper for more details. In any case if one of these tests fails, the quantum state is entangled. If the PSD condition fails for example, the SDP feasibility problem is unsolvable, the quantum state is entangled.