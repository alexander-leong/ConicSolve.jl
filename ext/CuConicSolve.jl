module CuConicSolve

using ConicSolve
using CUDA
using LinearAlgebra

if CUDA.functional()
    to_gpu_or_not_to_gpu(x) = CuArray(x)
else
    to_gpu_or_not_to_gpu(x) = x
end

function ConicSolve.get_device_array(::Int, obj)
    return CuArray{eltype(obj)}(obj)
end

end
