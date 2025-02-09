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

(i) Create a third order polynomial with all coefficients set to 1 like so:
```julia
s = Polynomial([1, 1, 1, 1], :t)
```

(ii) Create an SOS object and set a diagonal constraint on the Q matrix to readily resolve the coefficients of the third order polynomial
```julia
sos = SOS(s)
set_diagonal_Q_constraint(sos)
```

(iii) Set any other constraints, in this example, the initial position (to 0 at t = 0), final position (to 5 at t = 1), initial velocity (<= 5 at t = 0) and final velocity (<= 5 at t>0)
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

#### Get the solution

#### Further Comments