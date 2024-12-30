abstract type Cone end

function inv_hessian(H::AbstractArray)
    # TODO: figure out a better way!
    return inv(H)
end