function full_qr_solve(A, G, P, b_x, b_y, b_z, inv_W_b_z, program)
    if A == sparse(I, size(A)[1], size(A)[2])
        Q, R = A, A
    else
        # this solver uses the Julia (LAPACK based) QR factorization method
        # which is based on Householder reflections.
        Q, R = qr(A')
        Q = collect(Q)
    end
    Q_A = P + G' * G
    # solve for Q_1_x from third equation
    Q_1_x = inv(R)' * b_y
    Q_1 = @view Q[:, 1:length(Q_1_x)]
    Q_2 = @view Q[:, length(Q_1_x)+1:size(Q)[2]]
    b_1 = b_x + G' * inv_W_b_z
    b_2 = b_1
    Q_1_A = Q_1' * Q_A
    Q_21_A = Q_2' * Q_A
    # solve for Q_2_x from second equation
    Q_12_A = Q_21_A'
    Q_2_A = cholesky(Q_2' * Q_A * Q_2)
    A₁₁x₁ = Q_1_A * Q_1 * Q_1_x
    A₂₁x₁ = Q_21_A * Q_1 * Q_1_x
    inv_Q_2_A = inv(Q_2_A.U)
    L = inv_Q_2_A * inv_Q_2_A'
    Q_2_x = L * (Q_2' * b_2 - A₂₁x₁)
    # solve for y from first equation
    A₁₂x₂ = Q_1' * Q_12_A * Q_2_x
    y = R \ ((Q_1' * b_1) - A₁₁x₁ - A₁₂x₂)
    x = Q * [Q_1_x; Q_2_x]
    return x, y
end

function reduced_qr_solve(A, G, P, b_x, b_y, b_z, inv_W, program)
    # do full QR, eq. 55, coneprog.pdf
    Q, R = qr(A')
    R₁ = R
    Q = collect(Q)
    Q₁ = Q[:, 1:size(R)[2]]
    Q₂ = Q[:, size(R)[2]+1:size(Q)[2]]
    # do cholesky, eq. 55, coneprog.pdf
    Q_A = P + G' * G
    Q_A_QR = qr(Q_A)
    Q_A_QR_sqrt = Q₂' * Q_A_QR.R'
    L = cholesky(Q_A_QR_sqrt * Q_A_QR_sqrt').L
    Q₁ᵀ_x = inv(R₁)' * b_y
    # solve for x
    x = Q₁ * Q₁ᵀ_x
    b_2 = (b_x + G' * inv_W' * b_z)
    A₂₁ = Q₂' * Q_A * Q₁
    Q₂ᵀ_x = (L * L') \ ((Q₂' * b_2) - (A₂₁ * Q₁ᵀ_x))
    b_1 = b_2
    A₁ = Q₁' * Q_A
    # solve for y
    A₁₁x₁ = A₁ * Q₁ * Q₁ᵀ_x
    A₁₂x₂ = A₁ * Q₂ * Q₂ᵀ_x
    y = R \ ((Q₁' * b_1) - A₁₁x₁ - A₁₂x₂)
    return x, y
end

function qp_solve(program, G_scaled, inv_W_b_z, qr_solve="full")
    # page 498, 618 Boyd and Vandenberghe
    # page 29, coneprog.pdf
    A = program.A
    P = program.P
    b_x = @view program.KKT_b[program.inds_c]
    b_y = @view program.KKT_b[program.inds_b]
    b_z = @view program.KKT_b[program.inds_h]
    # TODO: cleanup
    if qr_solve == "full"
        x, y = full_qr_solve(A, G_scaled, P, b_x, b_y, b_z, inv_W_b_z, program)
    end
    x_vec = zeros(length(program.KKT_b))
    x_vec[program.inds_c] = x
    x_vec[program.inds_b] = y
    return x_vec
end
