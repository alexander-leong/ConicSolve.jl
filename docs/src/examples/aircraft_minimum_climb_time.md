# Aircraft Minimum Climb Time

## Introduction

## How to run the example

### Explanation

The following procedure is used to derive the QCQP (quadratic constrained quadratic program) optimization problem. This approach is typical when dealing with problems that involve Differential-Algebraic System of Equations (DAE) to obtain a nonlinear system that one solves using Interior Point methods.

1. Define the variables
2. Linearization of the vehicle dynamics equations
3. Trapezoidal collocation
4. Semidefinite relaxation
5. Set the initial and final conditions

#### Define the variables
Define the decision variables in a vector ``x`` consisting of the variables evaluated at the N collocation points:

```math
\boldsymbol{\gamma_0} = \begin{bmatrix}\gamma_0(t_0), \gamma_0(t_1),  ..., \gamma_0(t_N) \end{bmatrix}
```
```math
\boldsymbol{\gamma_1} = \begin{bmatrix}\gamma_1(t_0), \gamma_1(t_1),  ..., \gamma_1(t_N) \end{bmatrix}
```
```math
\boldsymbol{\gamma'} = \begin{bmatrix}\gamma'(t_0), \gamma'(t_1), ..., \gamma'(t_N) \end{bmatrix}
```

```math
\textbf{m} = \begin{bmatrix}m(t_0), m(t_1), ..., m(t_N) \end{bmatrix}
```
```math
\textbf{m'} = \begin{bmatrix}m'(t_0), m'(t_1), ..., m'(t_N) \end{bmatrix}
```

```math
\textbf{h'} = \begin{bmatrix}h'(t_0), h'(t_1), ..., h'(t_N) \end{bmatrix}
```

```math
\textbf{r'} = \begin{bmatrix}r'(t_0), r'(t_1), ..., r'(t_N) \end{bmatrix}
```

```math
\textbf{v'} = \begin{bmatrix}v'(t_0), v'(t_1), ..., v'(t_N) \end{bmatrix}
```

```math
\textbf{k} = \begin{bmatrix}k_0,  k_1, ..., k_N \end{bmatrix}
```

```math
\textbf{t} = \begin{bmatrix}t_0,  t_1, ..., t_N \end{bmatrix}
```
```math
x = \begin{bmatrix}
\boldsymbol{\alpha_0},
\boldsymbol{\alpha_1},
\textbf{m},
\boldsymbol{\gamma_0},
\boldsymbol{\gamma_1},
\textbf{h'},
\textbf{m'},
\textbf{r'},
\textbf{v'},
\boldsymbol{\gamma_0'},
\boldsymbol{\gamma_1'},
\boldsymbol{k},
\boldsymbol{p},
\boldsymbol{t},
c
\end{bmatrix}
```
where:

``\alpha_0 = cos(\alpha)``, ``\alpha_1 = sin(\alpha)``,

``\gamma_0 = cos(\gamma)`` and ``\gamma_1 = sin(\gamma)``

with the quadratic constraints defined as:

Constraints for ``\alpha`` and ``\gamma``:
```math
\begin{bmatrix}
\alpha_0 & \alpha_1
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\alpha_0 \\
\alpha_1
\end{bmatrix} = 1
```
```math
\begin{bmatrix}
\gamma_0 & \gamma_1
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\gamma_0 \\
\gamma_1
\end{bmatrix} = 1
```
#### Linearization of the vehicle dynamics equations
The dynamics equations are written in state-space form as follows:
```math
v' = \dfrac{T}{m}cos(\alpha) - \dfrac{D}{m} - gsin(\gamma) \implies
```
```math
\begin{bmatrix}
\boldsymbol{\alpha_0} &
\textbf{m} &
\textbf{v'} &
\boldsymbol{\gamma_1}
\end{bmatrix}
\begin{bmatrix}
0 & 1/2 & 0 & 0 \\
1/2 & 0 & 1/2 & (1/2)g \\
0 & 1/2 & 0 & 0 \\
0 & (1/2)g & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{\alpha_0} \\
\textbf{m} \\
\textbf{v'} \\
\boldsymbol{\gamma_1}
\end{bmatrix} +
\begin{bmatrix}
T & 0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{\alpha_0} \\
\textbf{m} \\
\textbf{v'} \\
\boldsymbol{\gamma_1}
\end{bmatrix} = -D
```
```math
\gamma' = \dfrac{T}{mv}sin(\alpha) + \dfrac{L}{mv} - \dfrac{gcos(\gamma)}{v} \implies
```
```math
\begin{bmatrix}
\alpha_1 & \textbf{m} & \textbf{p} & \textbf{v} & \boldsymbol{\gamma'} & \boldsymbol{\gamma_0}
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & (1/2)g \\
0 & 0 & 0 & 0 & 1/2 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1/2 & 0 & 0 & 0 \\
0 & (1/2)g & 0 & 0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\alpha_1 \\
\textbf{m} \\
\textbf{p} \\
\textbf{v} \\
\boldsymbol{\gamma'} \\
\boldsymbol{\gamma_0}
\end{bmatrix} +
\begin{bmatrix}
T & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\alpha_1 \\
\textbf{m} \\
\textbf{p} \\
\textbf{v} \\
\textbf{v'} \\
\boldsymbol{\gamma_1}
\end{bmatrix} = L
```
```math
\dfrac{h}{t} = vsin(\gamma) \implies
```
```math
\begin{bmatrix}
v & \textbf{h'} & \boldsymbol{\gamma_1}
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 1/2 \\
0 & 0 & 0 \\
1/2 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
v \\
\textbf{h'} \\
\boldsymbol{\gamma_1}
\end{bmatrix} -
\textbf{1}^T \textbf{h'} = 0
```
```math
\dfrac{r}{t} = vcos(\gamma) \implies
```
```math
\begin{bmatrix}
v & \textbf{r'} & \boldsymbol{\gamma_0}
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 1/2 \\
0 & 0 & 0 \\
1/2 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
v \\
\textbf{r'} \\
\boldsymbol{\gamma_0}
\end{bmatrix} -
\textbf{1}^T \textbf{r'} = 0
```
```math
m' = -\dfrac{T}{gI_{sp}}
```
which is constant with respect to time.
#### Trapezoidal Collocation
Using trapezoidal collocation, ``x(t+1)`` where ``x`` is some time-varying variable can be expressed as follows:
```math
x_{t+1} - x_t \approx \dfrac{1}{2}(k_t)(x'_{t+1} + x'_t) \implies
```
```math
\begin{bmatrix}
0 & -1 & 1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
k(t) \\
x(t) \\
x(t+1) \\
x'(t) \\
x'(t+1)
\end{bmatrix} -
\begin{bmatrix}
k(t) & x(t) & x(t+1) & x'(t) & x'(t+1)
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 0 & 1/4 & 1/4 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
1/4 & 0 & 0 & 0 & 0 \\
1/4 & 0 & 0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
k(t) \\
x(t) \\
x(t+1) \\
x'(t) \\
x'(t+1)
\end{bmatrix} = 0
```
where ``k_n`` is defined as:
```math
k_n = t_{n+1} - t_n
```

NOTE: in addition we add the necessary constraints so the collocation points are equi-distant.

Additionally we need constraints with respect to the collocation points (i.e. t_N >= t_{N-1} >= ... >= t_1 >= t_0) to ensure the values are non-decreasing.
```math
\begin{bmatrix}
-1 & 1 & 0 & ... & 0 & 0 \\
0 & -1 & 1 & ... & 0 & 0 \\
0 & 0 & -1 & ... & 0 & 0 \\
& & & \ddots & & \\
0 & 0 & 0 & ... & -1 & 1
\end{bmatrix}
\begin{bmatrix}
t_0 \\ t_1 \\ t_2 \\ ... \\ t_{N-1} \\ t_N
\end{bmatrix} \ge \textbf{0}
```
#### Semidefinite Relaxation
Solving QCQPs (Quadratic Constrained Quadratic Program) can be achieved by transforming the problem into an SDP of the form:
```math
\begin{aligned}
\text{minimize}\qquad & \langle M_0, 
\begin{bmatrix}
xx^T & x \\
x^T & 1
\end{bmatrix}\rangle \\
\text{subject to}\qquad & M_i \succeq 0
\end{aligned}
```
with ``M_i`` defined as:
```math
M_i = \begin{bmatrix}
A_i & b_i \\
b_i^T & c_i \\
\end{bmatrix}
```
Note that ``M_0`` and ``M_i`` are positive semidefinite which means the diagonal elements of these matrices must be nonnegative. The solver will take care of this for us.

#### Data Acquisition

#### Solve the problem

#### Get the solution

#### Further Comments