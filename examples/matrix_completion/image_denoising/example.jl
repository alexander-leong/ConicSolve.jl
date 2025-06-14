#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

""" Matrix completion example for image denoising
via Nuclear-norm minimization and SDP embedding lemma.
see Matrix Rank Minimization with Applications, Fazel (2002)
"""

include("./imageutils.jl")

using ConicSolve
using JLD

function denoise_image(img::Matrix{Float64},
                       noise::Matrix{Float64})
    sdp = SDP(img)
    set_values(sdp, noise)
    set_nonnegative_constraint(sdp)
    c = get_trace(sdp)
    set_objective(sdp, c)
    qp = sdp_to_qp(sdp)
    return qp
end

function preprocess_data()
    img = load_image()
    noisy_img = add_noise(img, 0.2)
    threshold = 0.99
    noise = detect_noise(noisy_img, threshold)

    img = convert(Array{Float64, 2}, img)
    noise = convert(Array{Float64, 2}, noise)
    return img, noise
end

function run_example()
    @info "Getting data"
    # img, noise = preprocess_data()
    data = load("/home/alexander/Documents/alexander_leong/ConicSolve.jl/data.jld")
    x_l, x_u, y_l, y_u = 107, 154, 225, 272
    img = data["img"]
    noise = data["noise"]
    img = get_block_matrix(img, x_l, y_l, x_u, y_u)
    noise = get_block_matrix(noise, x_l, y_l, x_u, y_u)
    @info "Constructing optimization problem"
    cone_qp = denoise_image(img, noise)
    solver = Solver(cone_qp)
    solver.max_iterations = 5
    status = run_solver(solver)
    return solver, status
    # x = get_solution(solver)
    @info "Done"
end

solver, status = run_example()
save("/home/alexander/Documents/alexander_leong/ConicSolve.jl/2d_reconstruction_2.jld", "kktsolution", solver.program.KKT_x, "s", solver.program.s)