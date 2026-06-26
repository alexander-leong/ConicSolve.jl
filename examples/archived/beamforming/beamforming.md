# Phased Array Diagram Matching

## Introduction

A common problem in phased array beamforming applications is setting the
weights (magnitude and phase) of each antenna to produce the desired diagram,
radiation pattern of the array.
If we make some simplifying assumptions, i.e. a planar antenna array, finding the weights can be expressed as a Conic LP problem,
cf. [Ben-Tal, Nemirovski, 2001, Lectures on Modern Convex Optimization](https://www2.isye.gatech.edu/~nemirovs/LMCOBookSIAM.pdf)
Consider the optimization problem below, in this case we'll assume a given direction δ.

```math
\begin{aligned}
\underset{z_1,...,z_N \in \mathbb{C}}{minimize}\qquad &
|Z_*(\delta) - \sum_{j=1}^Nz_jZ_j(\delta)|
\end{aligned}
```

## How to run the example

1. Ensure that you have ConicSolve installed. This can be installed as follows:
```julia
julia> ]
pkg> activate .
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

(i) We create a ConeQP object that represents the LP problem to solve `cone_qp = get_qp(diagrams, target)`.

(ii) We pass the ConeQP object to the solver `solver = Solver(cone_qp)`.

(iii) Then when we're ready we call `run_solver` passing the solver object `run_solver`(solver)`.

(iv) We can access the solution by accessing the primal solution from the solver `x = get_solution(solver)`.

#### Get the solution

#### Further Comments