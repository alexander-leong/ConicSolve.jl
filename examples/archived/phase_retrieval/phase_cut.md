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
where U is the decision variable with respect to the positive semidefinite cone and M is defined as:
```math
M = \textbf{diag}(b)(\textbf{I} - AA^\dagger)\textbf{diag}(b)
```
The matrix ``A`` is taken as the 1-D DFT matrix of size nxn where n is the length of the signal and ``A^\dagger`` is the Moore Penrose Pseudoinverse of ``A``.

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
\textbf{svec}(\textbf{diag}([1 & 0 & ... & 0]))^T \\
\textbf{svec}(\textbf{diag}([0 & 1 & ... & 0]))^T \\
\vdots \\
\textbf{svec}(\textbf{diag}([0 & 0 & ... & 1]))^T \\
\end{bmatrix}
u = \textbf{1}\\&
U \succeq 0
\end{aligned}
```
and using the fact that ``\textbf{Tr}(MU) = \textbf{svec}(M)^T\textbf{svec}(U)``.

Since ConicSolve.jl is a real-numbered solver, the Hermitian matrix M can be encoded as a real symmetric matrix, i.e.
```math
\begin{bmatrix}
\textbf{Re}(M) & -\textbf{Im}(M) \\
\textbf{Im}(M) & \textbf{Re}(M)
\end{bmatrix}
```

This example uses the "multiple random illumination filters" method for phase retrieval. In this case the matrix A consists of a stack of randomly generated circulant matrices defining each of the random illumination filters, i.e.
```math
A = \begin{bmatrix}
C_1 \\
C_2 \\
\vdots \\
C_m
\end{bmatrix}
```
where ``C_k`` is the $k^{th}$ Circulant matrix of size ``n`` x ``n``, ``n`` is the length of the signal being estimated.

#### Data Acquisition
This is a simple toy problem setup. No data has been imported in this example.

A simple sine wave function has been used as the signal we want to reconstruct.

#### Solve the problem

(i) We create a ConeQP object that represents the SDP problem to solve `cone_qp = get_qp()`.

(ii) We pass the ConeQP object to the solver `solver = Solver(cone_qp)`.

(iii) Then when we're ready we call `run_solver` passing the solver object `solver`.

(iv) We can access the solution by accessing the primal solution from the solver `u = get_solution(solver)`.

#### Get the solution

The output of `get_solution` is used by `reconstruct_signal` to evaluate the following expression giving the reconstructed signal.
```math
x = A^\dagger\textbf{diag}(b)u
```

Since this problem is an SDP relaxation and we know that U should be rank one (or close to it), a good way to check the tightness of the relaxation is to check the eigenvalues of ``U``.

If one of the eigenvalues is large and the rest are close to zero, we're good.

#### Further Comments

The aim is to get the reconstructed signal as close to the original input signal (the sine wave) generated. One way to measure the accuracy of the reconstruction is the Mean Squared Error (MSE) given by
```math
MSE = \dfrac{1}{n}\Sigma_{i=1}^n(b - Ax)^2
```

Sometimes the reconstruction may be poor because U is not close to rank one. We should notice the reconstruction of ``x`` improve by increasing the number of filters and sample points used over the signal.