# Total Variation Reconstruction

## Introduction

Problems in signal processing typically involve filtering, denoising and smoothing a signal. In this example we'll look at total variation reconstruction of a 1-D signal. We'll consider the optimization problem below:
```math
\begin{aligned}
\text{minimize}\qquad & \left \lVert \hat{x}-(x+v) \right \rVert _2 + \lambda \left \lVert D\hat{x} \right \rVert _1
\end{aligned}
```
This is a multi criterion (bicriterion) problem since there are two variables we want to optimize against. All multi criterion problems require a tradeoff and the parameter ``\lambda`` enables this. By increasing ``\lambda`` we encourage a solution that aims for ``\left \lVert D\hat{x} \right \rVert _1`` small. The converse is also true, reducing ``\lambda`` will encourage ``\left \lVert \hat{x}-(x+v) \right \rVert _2`` small.

In this example, ``\hat{x}`` is the decision variable and ``x+v`` is the 1-D input signal for some noise ``v`` which we want to denoise.

## How to run the example

1. Ensure that you have the necessary dependencies installed. These can be installed by executing the following commands:
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

We can write the optimization problem in conic form as follows:
```math
```

#### Data Acquisition
This is a simple toy problem setup. No data has been imported in this example.

``x+v`` is white noise generated using Julia's default pseudo-random number generator.

#### Solve the problem

#### Get the solution

#### Further Comments