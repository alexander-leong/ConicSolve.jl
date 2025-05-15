#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

# This file contains common subroutines for manipulating
# arrays and matrices for constructing an optimization problem.

"""
    svec(U)

### Output

Returns a vectorized representation of the square matrix U as
``(U_{11}, \\sqrt{2}U_{21}, ..., \\sqrt{2}U_{p1}, U_{22}, \\sqrt{2}U_{32}, ..., \\sqrt{2}U_{p2}, ..., U_{p-1,p-1}, \\sqrt{2}U_{p,p-1}, U_{pp})``
"""
function svec(X::AbstractArray{T}) where T<:Number
    N = size(X)
    if N[1] != N[2]
        throw(DimensionMismatch("X is not square"))
    end
    n = (N[1] * (N[1] + 1)) / 2
    x = zeros(T, (Int(n)))
    k = 1
    for j in 1:N[1]
        for i = j:N[1]
            if i != j
                x[k] = sqrt(2) * X[i, j]
            else
                x[k] = X[i, j]
            end
            k = k + 1
        end
    end
    return x
end

function mat_to_lower_vec(X)
    N = size(X)
    if N[1] != N[2]
        throw(DimensionMismatch("X is not square"))
    end
    n = (N[1] * (N[1] + 1)) / 2
    x = zeros((Int(n)))
    k = 1
    for j in 1:N[1]
        for i = j:N[1]
            x[k] = X[i, j]
            k = k + 1
        end
    end
    return x
end

"""
    mat(u)

Performs the inverse operation of vec(U), i.e. mat(vec(U)) = U

### Output
Returns a matrix representation of the vector u.
```math
\\begin{bmatrix}
    u_{1}            & u_{2}/\\sqrt{2}     & u_{3}/\\sqrt{2}   & \\dots  & u_{n}/\\sqrt{2}     \\\\
    u_{2}/\\sqrt{2}  & u_{p+1}             & u_{23}/\\sqrt{2}  & \\dots  & u_{2p-1}/\\sqrt{2}  \\\\
    \\vdots          & \\vdots             & \\vdots           &         & \\vdots             \\\\
    u_{p}/\\sqrt{2}  & u_{2p-1}/\\sqrt{2}  & u_{d3}/\\sqrt{2}  & \\dots  & u_{p(p+1)/2}
\\end{bmatrix}
```
"""
function mat(x::AbstractArray{T}) where T<:Number
    n = 0
    m = 1
    k = 0
    while n <= length(x)
        n = n + 1
        k = k + 1
        if k > m
            m = m + 1
            k = 1
        end
    end
    if n < m
        throw(DimensionMismatch("x is not triangular"))
    end
    n = m - 1
    X = zeros(T, (n, n))
    k = 1
    for j = 1:n
        for i = j:n
            if i != j
                X[i, j] = x[k] / sqrt(2)
                X[j, i] = x[k] / sqrt(2)
            else
                X[i, j] = x[k]
            end
            k = k + 1
        end
    end
    return Hermitian(X)
end

"""
    apply_cart_on_f(f, args...)

### Output

Returns an array of f(x...) passing each value x of the cartesian product args...
i.e. example input
```julia
a = [0, 1]
b = [3, 2]
vals = apply_cart_on_f(f, a, b)
```
yields
vals = [f(0, 3), f(0, 2), f(1, 3), f(1, 2)]
"""
function apply_cart_on_f(f, args...)
    vals = map(z -> f(z...), reduce(vcat, Iterators.product(args...)))
    return vals
end

function flatten_2d(mask_idx, nCols)
    return map(x -> nCols * (x[1] - 1) + x[2], mask_idx)
end

function flatten_2d_idx(nCols, X_idx)
    idx = flatten_2d(X_idx, nCols)
    return idx
end

"""
    lower_triangular_from_2d_idx(n, idx)

Converts 2d index values of a square matrix of size n x n.

### Output

Return the index values of a vectorized lower triangular matrix.
"""
function lower_triangular_from_2d_idx(n, idx)
    count = 0
    sizes = zeros((n))
    for i in 2:n
        count += n - i + 2
        sizes[i] = count
    end
    return map(x -> Int(sizes[x[2]] + x[1] - x[2] + 1), idx)
end

function ohe_from_2d_mask(
    A, 
    idx, 
    mask, 
    nCols,
    offsetX,
    offsetY
)
    function filterFunc(x)
        return mask[x[1] - offsetX, x[2] - offsetY] == 0
    end
    mask_idx = filter(filterFunc, idx)
    mask_img_idx = flatten_2d(mask_idx, nCols)
    A_idx = hcat(mask_img_idx, mask_img_idx)
    A_idx = [CartesianIndex(x[1], x[2]) for x in eachrow(A_idx)]
    A[A_idx] .= 1
end

"""
    dropzero_rows(X)

Removes row of zeros from a given matrix X.
"""
function dropzero_rows(X)
    non_zero_row = map(x -> x != zeros(length(x)), eachrow(X))
    inds = findall(non_zero_row)
    new_X = X[inds, :]
    return new_X, inds
end

"""
    get_block_matrix(X, x_l, y_l, x_u, y_u)

Get block matrix defined by the row and column range. \\
x\\_l <= i <= x\\_u \\
y\\_l <= j <= y\\_u
"""
function get_block_matrix(X, x_l, y_l, x_u, y_u)
    return X[x_l:x_u, y_l:y_u]
end

function get_img_idx_in_A(A_idx, img)
    idx = A_idx[1:size(img)[1], end-size(img)[2]+1:end][:]
    return idx
end

function get_img_transposed_idx_in_A(A_idx, img)
    idx = A_idx[end-size(img)[2]+1:end, 1:size(img)[1]][:]
    return idx
end

"""
    get_mask(cond, data)

Creates a mask given a predicate function, cond
against the data matrix.
"""
function get_mask(cond, data)
    return map(cond, data)
end

"""
    get_off_diagonal_idx(n)

Get vectorized lower triangular off diagonal indices.
For example, the following entries returned:
idx = [2, 3, 4, 6, 7, 9] correspond to the matrix of n=4:
[1 2 3 4; 2 5 6 7; 3 6 8 9; 4 7 9 10]
"""
function get_off_diagonal_idx(n)
    idx = []
    for j = 1:n
        for i = j+1:n
            push!(idx, CartesianIndex((i, j)))
        end
    end
    mask = ones(Bool, (n, n))
    mask[diagind(mask)] .= false
    idx = filter(x -> mask[x[1], x[2]], idx)
    idx = lower_triangular_from_2d_idx(n, idx)
    return idx
end

"""
    get_diagonal_idx(n)

Get vectorized diagonal indices.
For example, the following entries returned:
idx = [1, 5, 8, 10] correspond to the matrix of n=4:
[1 2 3 4; 2 5 6 7; 4 7 9 10]
"""
function get_diagonal_idx(n)
    idx = []
    for i = 1:n
        push!(idx, CartesianIndex((i, i)))
    end
    idx = lower_triangular_from_2d_idx(n, idx)
    return idx
end

"""
    get_triangular_idx(A_idx, mask)

Get vectorized lower triangular indices given a 2d mask.
"""
function get_triangular_idx(A_idx, mask)
    b_idx = vec(get_img_transposed_idx_in_A(A_idx, mask))
    n = size(mask)[1] + size(mask)[2]
    offset_X = n - size(mask)[2]
    # filter X with mask
    # X = [[., X],
    #     [X^T, .]]
    b_idx = filter(x -> mask'[x[1] - offset_X, x[2]] == 1, b_idx)
    b_idx = lower_triangular_from_2d_idx(n, b_idx)
    return b_idx
end

export svec
export mat
export mat_to_lower_vec
export lower_triangular_from_2d_idx
export dropzero_rows
export get_block_matrix
export get_mask
export get_off_diagonal_idx
export get_diagonal_idx
export get_triangular_idx