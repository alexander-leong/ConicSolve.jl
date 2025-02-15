#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

"""
    SDP

Represents a Semidefinite Program (SDP). \\
i.e. An optimization of the form
```math
\\begin{aligned}
\\text{minimize}\\qquad &
c^Tx \\\\
\\text{subject to}\\qquad &
Ax = b \\\\
& x \\succeq 0
\\end{aligned}
``` \\
NOTE: Linear Matrix Inequalities not supported yet.
"""
mutable struct SDP
    A
    G
    P
    b
    c
    h
    cones::Vector{Cone}
    idx
    num_rows
    num_cols
    X_1
    X_1_idx

    # X = [X_2 X_1; X_1' X_4]
    @doc"""
        SDP(X_1)
    
    Constructs an SDP optimization problem. \\
    NOTE: only elements of X_1 can be set for now. \\
    The SDP matrix is given by
    ```math
    X = \\begin{bmatrix}
        X_2  & X_1 \\\\
        X_1' & X_4
    \\end{bmatrix}
    ```

    # Parameters:
    * `X_1`: The block matrix X_1 of the SDP matrix X
    """
    function SDP(X_1)
        sdp = new()
        sdp.num_rows = size(X_1)[1] + size(X_1)[2]
        sdp.num_cols = sdp.num_rows
        sdp.cones = []
        push!(sdp.cones, PSDCone(sdp.num_rows))
        sdp.idx = CartesianIndices((sdp.num_rows, sdp.num_cols))
        sdp.X_1 = X_1
        sdp.X_1_idx = sdp.idx[1:size(X_1)[1], end - size(X_1)[2]:end][:]
        return sdp
    end
end

"""
    set_nonnegative_constraint(sdp)

Impose a nonnegative constraint on all values of the SDP matrix.
"""
function set_nonnegative_constraint(sdp::SDP)
    p = Int((sdp.num_rows * (sdp.num_rows + 1)) / 2)
    push!(sdp.cones, NonNegativeOrthant(p))
    G = get_off_diag_nonnegative_constraint(sdp.A)
    sdp.G = vcat(sdp.G, G)
end

"""
    set_objective(sdp, c)

Set the objective function of the SDP, which is the function
``\\langle c, x \\rangle``
"""
function set_objective(sdp::SDP, c)
    sdp.c = c
end

"""
    set_values(sdp, mask)

Set linear equalities of the SDP constraints where values
of the SDP matrix ``X`` are determined by ``X_1`` and the mask.
```julia
if mask[i, j]
    X_1[i, j] = X_1[i, j]
end
```
"""
function set_values(sdp::SDP, mask)
    # set psd constraint
    A, G, b = set_off_diag_constraint(sdp, sdp.X_1, mask)
    sdp.A = A
    sdp.G = G
    sdp.b = b
end

"""
    sdp_to_qp(sdp)

Get the Cone QP object representing the SDP problem
"""
function sdp_to_qp(sdp::SDP)
    sdp.P = zeros((size(sdp.A)[2], size(sdp.A)[2]))
    sdp.h = zeros(size(sdp.G)[1])
    cone_qp = ConeQP(sdp.A, sdp.G, sdp.P, sdp.b, sdp.c, sdp.h, sdp.cones)
    return cone_qp
end

"""
    get_X1(sdp, x)

Returns the values of X1 in the SDP matrix
```math
X = \\begin{bmatrix}
    X_2 & X_1^T \\\\
    X_1 & X_4
\\end{bmatrix}
```
"""
function get_X1(sdp::SDP, x)
    X1_idx = vec(get_img_transposed_idx_in_A(sdp.idx, x))
    X1 = x[X1_idx]
    return X1
end

"""
    set_b_from_data(sdp, data)

Set the values of the vector b in ConeQP based on
the data matrix.
"""
function set_b_from_data(sdp::SDP, data)
    dim = size(sdp.idx)
    X_size = Int((dim[1] * (dim[1] + 1)) / 2)
    b = zeros(X_size)
    b_idx = vec(get_img_transposed_idx_in_A(sdp.idx, data))
    n = size(data)[1] + size(data)[2]
    b_idx = lower_triangular_from_2d_idx(n, b_idx)
    b[b_idx] = vec(data)
    return b_idx, b
end

"""
    set_off_diag_constraint(sdp, data, mask)

Set the off diagonal entries in the SDP matrix.
i.e. [Y data; data' Z]
The mask is a boolean matrix that corresponds to which elements
of the data matrix to keep.

### Output

The precomputed matrices A, G and the vector b to pass
as arguments to ConeQP.
"""
function set_off_diag_constraint(sdp::SDP, data, mask)
    dim = (sdp.num_rows, sdp.num_cols)
    X_size = Int((dim[1] * (dim[1] + 1)) / 2)

    # construct A from [Y data; data' Z]
    A = Matrix(1.0I, X_size, X_size)
    
    # set A, b as values of non noise image
    b_idx, b = set_b_from_data(sdp, data)
    noise_idx = get_triangular_idx(sdp.idx, mask)
    noise_idx = map(x -> CartesianIndex(x, x), noise_idx)
    A[noise_idx] .= 0
    A = A[b_idx, :]
    
    # let initialization deal with diagonal dominance
    b = svec(mat(b))

    # omit zero rows in A and respective entries in b
    A, inds = dropzero_rows(A)
    b = b[inds]
    
    G = -Matrix(1.0I, size(A)[2], size(A)[2])

    return A, G, b
end

"""
    get_trace(sdp)

Used to express the trace of the decision variable ``X``
in the objective function.

### Output

The vector c to pass as argument to ConeQP.
"""
function get_trace(sdp::SDP)
    dim = (sdp.num_rows, sdp.num_cols)
    c_ones = map(x -> CartesianIndex(x[1], x[1]), 1:dim[1])
    c_ones = lower_triangular_from_2d_idx(dim[1], c_ones)
    c = zeros(Int((dim[1] * (dim[1] + 1)) / 2))
    c[c_ones] .= 1
    return c
end

"""
    get_off_diag_nonnegative_constraint(A)

Set a nonnegative constraint on the off diagonal entries
of the SDP matrix.

### Output

The matrix G to pass as argument to ConeQP.
"""
function get_off_diag_nonnegative_constraint(A)
    G = -Matrix(1.0I, size(A)[2], size(A)[2])
    return G
end

export SDP
export set_nonnegative_constraint
export set_objective
export set_values
export sdp_to_qp
export get_X1
export set_b_from_data
export set_off_diag_constraint
export get_trace
export get_off_diag_nonnegative_constraint