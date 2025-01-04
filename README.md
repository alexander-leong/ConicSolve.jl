# ConicSolve.jl

ConicSolve.jl is an Interior Point based constrained optimization solver based on the paper [Vandenberghe, 2010, The CVXOPT linear and quadratic cone program solvers](https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf). It can be used to solve several problem classes including LP (Linear Program), QP (Quadratic Program) and SDP (Semidefinite Program).

## Installation

To use ConicSolve.jl, install [Julia](https://julialang.org/downloads/), then at the Julia REPL, type:

```julia
using Pkg
Pkg.add("ConicSolve")
using ConicSolve
```

# Documentation

Please refer to the package documentation at [Package Documentation](https://github.com/alexander-leong/ConicSolve.jl/blob/v0.0.1/docs/src/index.md)

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