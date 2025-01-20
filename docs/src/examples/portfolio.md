# Portfolio Optimization

## Introduction

The portfolio optimization example as described in [Boyd, Vandenberghe, 2009, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) can be formulated as a Conic SOCP program as follows.

```math
\begin{aligned}
\text{maximize}\qquad &
\mathbb{E}(p)^Tx \\
\text{subject to}\qquad &
\mathbb{E}(p)^Tx + \Phi^{-1}(\beta)\left \lVert \Sigma^{1/2}x \right \rVert _2 \ge \alpha \\
& x \ge 0 \\
& \textbf{1}^Tx = 1
\end{aligned}
```

## How to run the example

1. Ensure that you have the necessary dependencies installed. These can be installed by executing the following commands:
```julia
julia> ]
pkg> activate .
pkg> add CSV
pkg> add DataFrames
pkg> add Distributions

pkg> add ConicSolve
julia> exit()
```

2. Run the example from the command line
```bash
julia example.jl
```

### Explanation

The portfolio optimization example is an example for maximizing expected return given a set of stocks subject to some risk tolerance. A table of stock ticks can be found in portfolio.csv, i.e. the daily close price of seven random stocks.

#### Data Acquisition

CSV.jl is used to load the file portfolio.csv into a DataFrame object (using DataFrames.jl). The data frame is then converted to a matrix for the solver.

#### Solve the problem

The problem parameters are calculated using Distributions.jl and Statistics.jl to get the expected return and price change variance. The loss risk constraint contains two tuning parameters ``\beta \le 0.5`` (based on the inv. cdf of the std. normal distribution) to control the level of risk and ``\alpha``, the loss amount for the level of risk.

#### Get the solution

For this problem setting the parameter ``\eta`` to nothing (i.e. ``\sigma``) may give better results.

The solution is x which can be obtained as below:
``x \in [0, 1]`` is the optimal weight of each stock in the portfolio.

```julia
x = get_solution(solver)
```