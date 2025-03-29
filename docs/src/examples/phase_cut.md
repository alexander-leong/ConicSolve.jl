# Phase Retrieval

## Introduction

[PhaseCut](https://www.di.ens.fr/~mallat/papiers/WaldspurgerMathProgr.pdf) is an algorithm for solving the phase retrieval problem that has applications in X-ray crystallography, transmission electron microscopy and coherent diffractive imaging for example.

We'll consider the task of recovering the phase of a 1-D signal. The optimization problem being solved is written as:
```math
\begin{aligned}
\text{minimize}\qquad & \textbf{Tr}(UM) \\
\text{subject to}\qquad &
\textbf{diag}(U) = \textbf{1}\\&
U \succeq 0\\&
\end{aligned}
```
where U is the decision variable and M is defined as:
```math
M = \textbf{diag}(b)(\textbf{I} - AA^\dagger)\textbf{diag}(b)
```
The matrix ``A`` is taken as the 1-D DFT matrix of size nxn where n is the length of the signal and ``A^\dagger`` is the complex conjugate of ``A``.

## How to run the example

1. Ensure that you have the necessary dependencies installed. These can be installed by executing the following commands:
```julia
julia> ]
pkg> activate .
pkg> add FFTW

pkg> add ConicSolve
julia> exit()
```

### Explanation

We can write the optimization problem in conic form as follows:
```math
\begin{aligned}
\text{minimize}\qquad & \textbf{svec}(M)^T\textbf{svec}(U) \\
\text{subject to}\qquad &
\begin{bmatrix}
\textbf{svec}(\textbf{diag}([1 & 0 & ... 0]))^T \\
\textbf{svec}(\textbf{diag}([0 & 1 & ... 0]))^T \\
\textbf{svec}(\textbf{diag}([0 & 0 & ... 1]))^T \\
\end{bmatrix}
u = \textbf{1}\\&
U \succeq 0
\end{aligned}
```
where ``u = \textbf{svec}(U)`` and using the fact that ``\textbf{Tr}(MU) = \textbf{svec}(M)^T\textbf{svec}(U)``.

#### Data Acquisition
This is a simple toy problem setup. No data has been imported in this example.

White noise was generated using Julia's default pseudo-random number generator.

#### Solve the problem

(i) We create a ConeQP object that represents the SDP problem to solve `cone_qp = get_qp()`.

(ii) We pass the ConeQP object to the solver `solver = Solver(cone_qp)`.

(iii) Then when we're ready we call optimize! passing the solver object `optimize!(solver)`.

(iv) We can access the solution by accessing the primal solution from the solver `x = get_solution(solver)`.

#### Get the solution

#### Further Comments