function svec(X::AbstractArray)
    N = size(X)
    if N[1] != N[2]
        throw(DimensionMismatch("X is not square"))
    end
    n = (N[1] * (N[1] + 1)) / 2
    x = zeros((Int(n)))
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

function mat(x::AbstractArray)
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
    X = zeros((n, n))
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
    return X
end

function flatten_2d(mask_idx, nCols)
    return map(x -> nCols * (x[1] - 1) + x[2], mask_idx)
end

function flatten_2d_idx(nCols, X_idx)
    idx = flatten_2d(X_idx, nCols)
    return idx
end

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

function dropzero_rows(X)
    non_zero_row = map(x -> sum(x) > 0, eachrow(X))
    inds = findall(non_zero_row)
    new_X = X[inds, :]
    return new_X, inds
end

function get_image_patch(img, x_l, y_l, x_u, y_u)
    return img[x_l:x_u, y_l:y_u]
end

function get_img_idx_in_A(A_idx, img)
    idx = A_idx[1:size(img)[1], end-size(img)[2]+1:end][:]
    return idx
end

function get_img_transposed_idx_in_A(A_idx, img)
    idx = A_idx[end-size(img)[2]+1:end, 1:size(img)[1]][:]
    return idx
end

function get_mask(cond, data)
    return map(cond, data)
end

function set_b_from_img(A, A_idx, img)
    b = zeros(size(A)[2])
    b_idx = vec(get_img_transposed_idx_in_A(A_idx, img))
    b_idx = lower_triangular_from_2d_idx(size(img)[1] + size(img)[2], b_idx)
    b[b_idx] = vec(img)
    return b_idx, b
end

function filter_triangular_idx(A_idx, noise)
    b_idx = vec(get_img_transposed_idx_in_A(A_idx, noise))
    b_idx = filter(x -> noise[x[1] - size(noise)[1], x[2]] == 1, b_idx)
    b_idx = lower_triangular_from_2d_idx(size(noise)[1] + size(noise)[2], b_idx)
    return b_idx
end