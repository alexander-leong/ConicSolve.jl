#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

function Base.:+(lhs::L1Norm, rhs::L2Norm)
end

function Base.:+(lhs::L2Norm, rhs::L1Norm)
end

function Base.:-(lhs::L1Norm, rhs::L2Norm)
end

function Base.:-(lhs::L2Norm, rhs::L1Norm)
end