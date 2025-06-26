# ConicSolve.jl

ConicSolve.jl is an Interior Point based constrained optimization solver based on the paper [Vandenberghe, 2010, The CVXOPT linear and quadratic cone program solvers](https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf). It can be used to solve several problem classes including LP (Linear Program), QP (Quadratic Program) and SDP (Semidefinite Program).

## Installation

There are two ways to install ConicSolve.jl
- As a Julia Package, see [Package Installation](#Package-Installation)
- As a standalone binary executable, see [Standalone Installation](#Standalone-Installation)

### Package Installation

To use ConicSolve.jl, install [Julia](https://julialang.org/downloads/), then at the Julia REPL, type:

```julia
using Pkg
Pkg.add("ConicSolve")
using ConicSolve
```

### Standalone Installation

At present, ConicSolve.jl (without CUDA support) can be run as a binary executable on Ubuntu x86-64 systems.

Download the tar.gz archive from the project's GitHub build artifacts. After extracting the archive, there is a bin directory containing the ConicSolve binary executable.

See [File based API](#File-based-API) for details on solving an optimization problem using the binary executable.

## File based API

The matrices, ``A``, ``G``, ``P``, and vectors, ``b``, ``c``, ``h`` for the optimization problem
```math
\begin{aligned}
\text{minimize}\qquad &
x^TPx \\
\text{subject to}\qquad &
h - Gx \succeq 0
\end{aligned}
```
where
```math
\begin{aligned}
G = \begin{bmatrix}
2 & 1 \\
-0.2 & -1 \\
-0.533333 & -1
\end{bmatrix} \qquad
P = \begin{bmatrix}
0.333333 & 0 \\
0 & 0.25 \\
\end{bmatrix} \qquad
h = \begin{bmatrix}
10 \\
10 \\
-3.75
\end{bmatrix}
\end{aligned}
```
can be defined in the file ``problem.txt`` (for example) below.

__WARNING__: Newline between the definitions of ``G``, ``P``, ``c``, ``h``, ``cones``, ``Solver`` is important!
```
ConeQP
G
2, 1
-0.2, -1
-0.533333, -1

P
0.333333, 0
0, 0.25

c
0, 0

h
10, 10, -3.75

cones
NonNegativeOrthant, 3

Solver
max_iterations, 10

```

The vector ``x`` is nonnegative, this is expressed in the cones section of the file as ``NonNegativeOrthant, 3`` where ``3`` is the number of elements (since ``x`` is length ``3``).

Solve the problem by executing the following command. In bash for example
```bash
./conicsolve/bin/ConicSolve ./output.txt ./problem.txt
```
where the solver output is written to ``output.txt`` and the file defining the optimization problem is ``problem.txt``

# Documentation

Please refer to the package documentation at [Package Documentation](https://alexander-leong.github.io/ConicSolve.jl/dev/)

# Reference Material

<a href="https://github.com/alexander-leong/ConicSolve.jl/tree/main/docs/src/assets/ConicSolve.pdf">Project Poster

# License
MIT License

Copyright (c) 2025 Alexander Leong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.