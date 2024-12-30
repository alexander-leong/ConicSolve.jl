mutable struct SDP
    idx
    num_rows
    num_cols
    X_1
    X_1_idx

    # X = [X_2 X_1; X_1' X_4]
    function SDP(X_1)
        sdp = new()
        sdp.num_rows = size(X_1)[1] + size(X_1)[2]
        sdp.num_cols = sdp.num_rows
        sdp.idx = CartesianIndices((sdp.num_rows, sdp.num_cols))
        sdp.X_1 = X_1
        sdp.X_1_idx = sdp.idx[1:size(X_1)[1], end - size(X_1)[2]:end][:]
        return sdp
    end
end

function set_off_diag_constraint(dim, img, noise)
    X_size = Int((dim[1] * (dim[1] + 1)) / 2)

    # construct A from [Y img; img' Z]
    A_idx = CartesianIndices(dim)
    A = Matrix(1.0I, X_size, X_size)
    
    # set A, b as values of non noise image
    # set_A_from_img!(A, A_idx, img, dim[2], noise)
    b_idx, b = set_b_from_img(A, A_idx, img)
    noise_idx = filter_triangular_idx(A_idx, noise)
    noise_idx = map(x -> CartesianIndex(x[1], x[1]), noise_idx)
    A[noise_idx] .= 0
    A = A[b_idx, :]
    # ensure b is PSD constraint
    # ensure Diag(mat(b)) large enough (i.e. diagonally dominant) to ensure PSD condition
    # b = svec(mat(b) + 1e3 * I)
    
    # let initialization deal with diagonal dominance
    b = svec(mat(b))

    # omit zero rows in A and respective entries in b
    A, inds = dropzero_rows(A)
    b = b[inds]
    
    G = -Matrix(1.0I, size(A)[2], size(A)[2])

    return A, G, b
end

function set_off_diag_nonnegative_constraint(sdp::SDP, A)
    dim = (sdp.num_rows, sdp.num_cols)
    G = -Matrix(1.0I, size(A)[2], size(A)[2])
    c_ones = map(x -> CartesianIndex(x[1], x[1]), 1:dim[1])
    c_ones = lower_triangular_from_2d_idx(dim[1], c_ones)
    c = zeros(Int((dim[1] * (dim[1] + 1)) / 2))
    c[c_ones] .= 1
    return G, c
end

function getSDPVariablePrimal(
    KKT_x, 
    img_idx
)
    idx = flatten_2d_idx(size(KKT_x)[2], img_idx)
    return KKT_x[idx]
end