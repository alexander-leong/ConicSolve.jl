# Trajectory Optimization

## Introduction

Optimizing a trajectory is a typical problem in motion planning applications such as robotics. Representing a trajectory using polynomials allows one to express an optimization problem solved using sum-of-squares programming. Since a trajectory in this example uses univariate polynomials, a sum-of-squares optimization problem will be equivalent to solving a semidefinite program which we know how to solve, cf. [Blekherman, Parrilo, Thomas, Semidefinite Optimization and Convex Algebraic Geometry](https://www.mit.edu/~parrilo/sdocag/MO13-Blekherman-Parrilo-Thomas.pdf).

The optimization problem to be solved will be mathematically expressed as follows:

```math
\begin{aligned}
\text{minimize}\qquad &
p^{'''}(t) \\
\text{subject to}\qquad &
p(t) \text{ is SOS}\\ &
p(0) = 0\\ &
p(1) = 5\\ &
p^{'}(0) \le 1.2\\ &
p^{'}(1) \le 2.5
\end{aligned}
```

where ``p(t) = a_nt^n + a_{n-1}t^{n-1} + ... + a_1t^1 + a_0``

In other words, we want to minimize the jerk (third derivative of position) such that the square of the coefficients are SOS and the initial position, final position, initial velocity and final velocity are given.

## How to run the example

1. Ensure that you have the necessary dependencies installed. These can be installed by executing the following commands:
```julia
julia> ]
pkg> activate .
pkg> add Polynomials

pkg> add ConicSolve
julia> exit()
```

2. Run the example from the command line
```bash
julia example.jl
```

### Explanation

#### Data Acquisition
This is a simple toy problem setup. No data has been imported in this example.

#### Solve the problem

(i) Create a fourth order polynomial with all coefficients set to 1 like so:
```julia
s = Polynomial([1, 1, 1, 1, 1], :t)
```

(ii) Create an SOS object and set a diagonal constraint on the Q matrix to readily resolve the coefficients of the fourth order polynomial
```julia
sos = SOS(s)
```

(iii) Set any other constraints, in this example, the initial position (to 0 at t = 0), final position (to 5 at t = 1), initial velocity (<= 1.2 at t = 0) and final velocity (<= 2.5 at t = 1)
```julia
v_sos = derivative(s)
p = evaluate_monomials(s, 0)
add_polynomial_equality_constraint(sos, 0, p)
p = evaluate_monomials(s, 1)
add_polynomial_equality_constraint(sos, 5, p)
p = evaluate_monomials(v_sos, 0)
add_polynomial_inequality_constraint(sos, 1.2, p)
p = evaluate_monomials(v_sos, 1)
add_polynomial_inequality_constraint(sos, 2.5, p)
```

(iv) Set the objective by setting the vector c
```julia
c = zeros(size(sos.A)[2])
c[1] = 1
set_objective(sos, c)
```

(v) Get QP and solve
```julia
cone_qp = get_qp(sos)
solver = Solver(cone_qp)
optimize!(solver)
```

(vi) Get solution from solver, i.e. values of Q
```julia
x = get_solution(solver)
```

(vii) Reconstruct polynomial (see below)

#### Get the solution

The solution is given by the value of the coefficients for each of the Q values that constitute the SOS decomposition. Therefore to recover the coefficients of the monomial terms of the polynomial of interest is just to sum the coefficients of the same order. i.e.
```math
\begin{align*}
a_0 &= Q_{55} \\
a_1 &= Q_{54} \\
a_2 &= Q_{44} + Q_{53} \\
a_3 &= Q_{43} + Q_{52} \\
a_4 &= Q_{33} + Q_{42} + Q_{51}
\end{align*}
```
for a 4th order polynomial with the following quadratic form
```math
p(t) = \begin{bmatrix} t^4 & t^3 & t^2 & t^1 & 1 \end{bmatrix}
\begin{bmatrix}
Q_{11} & Q_{12} & Q_{13} & Q_{14} & Q_{15} \\
Q_{21} & Q_{22} & Q_{23} & Q_{24} & Q_{25} \\
Q_{31} & Q_{32} & Q_{33} & Q_{34} & Q_{35} \\
Q_{41} & Q_{42} & Q_{43} & Q_{44} & Q_{45} \\
Q_{51} & Q_{52} & Q_{53} & Q_{54} & Q_{55}
\end{bmatrix}
\begin{bmatrix} t^4 \\ t^3 \\ t^2 \\ t^1 \\ 1 \end{bmatrix}
```

#### Further Comments

We can impose further constraints such as polyhedral constraints for collision avoidance (see [MotionPlanningOptimization.jl](https://github.com/alexander-leong/MotionPlanningOptimization.jl)).