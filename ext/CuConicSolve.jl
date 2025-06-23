module CuConicSolve

using ConicSolve
using CUDA
using LinearAlgebra

if CUDA.functional()
    to_gpu_or_not_to_gpu(x) = CuArray(x)
else
    to_gpu_or_not_to_gpu(x) = x
end

function ConicSolve.get_array(device, obj)
    if device == GPU
        return CuArray{eltype(obj)}(obj)
    end
    return obj
end

end
