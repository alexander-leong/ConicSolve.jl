using CUDA
if CUDA.functional()
    to_gpu_or_not_to_gpu(x) = CuArray(x)
else
    to_gpu_or_not_to_gpu(x) = x
end

function qr_chol_cpu_to_gpu(Q, Q_A, Q_1, Q_2, R, b_y, b_1, b_2)
    Q = CuArray(Q)
    Q_A = CuArray(Q_A)
    Q_1 = CuArray(Q_1)
    Q_2 = CuArray(Q_2)
    R = CuArray(R)
    b_y = CuArray(b_y)
    b_1 = CuArray(b_1)
    b_2 = CuArray(b_2)
    return Q, Q_A, Q_1, Q_2, R, b_y, b_1, b_2
end

function qr_chol_gpu_to_cpu(x_cpu, x_gpu, y_cpu, y_gpu)
    copyto!(x_cpu, x_gpu)
    copyto!(y_cpu, y_gpu)
end