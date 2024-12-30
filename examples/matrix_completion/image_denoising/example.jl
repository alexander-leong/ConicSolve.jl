# Matrix completion example for image denoising
# via Nuclear-norm minimization
# see Matrix Rank Minimization with Applications, Fazel (2002)
# Alexander Leong 2024

include("../../../src/cones/nonneg.jl")
include("../../../src/cones/psdcone.jl")
include("../../../src/arrayutils.jl")
include("../../../src/imageutils.jl")
include("../../../src/models/sdp.jl")
include("../../../src/solver.jl")

using JLD
using SparseArrays

function solve(cone_qp::ConeQP)
    solver = Solver(cone_qp)

    # optimize for denoised image
    optimize!(solver)
    
    # return denoised image
    # img_idx = flatten_2d_idx(sdp.num_cols, sdp.X_1_idx)
    # KKT_x = getVariablePrimal(solver)
    # return getSDPVariablePrimal(KKT_x, img_idx)
end

function denoise_image(img::Matrix{Float64},
                       noise::Matrix{Float64})
    # set parameters to solve an SDP problem
    sdp = SDP(img)
    dim = (sdp.num_rows, sdp.num_cols)

    # set psd constraint
    A, G_1, b = set_off_diag_constraint(dim, img, noise)

    # set nonnegativity constraint
    G_2, c = set_off_diag_nonnegative_constraint(sdp, A)

    G = [G_1; G_2]
    P = zeros((size(A)[2], size(A)[2]))
    h = zeros(size(G)[1])
    
    # construct optimization problem
    p = Int((sdp.num_rows * (sdp.num_rows + 1)) / 2)
    cones::Vector{Cone} = []
    push!(cones, PSDCone(sdp.num_rows))
    push!(cones, NonNegativeOrthant(p))
    cone_qp = ConeQP(A, G, P, b, c, h, cones)
    cone_qp.cones_inds = [0, size(G_1)[1],
    size(G_1)[1] + size(G_2)[1]]
    return cone_qp, sdp
end

function preprocess_data()
    img = load_image()
    noisy_img = add_noise(img, 0.2)
    threshold = 0.99
    noise = detect_noise(noisy_img, threshold)

    img = convert(Array{Float64, 2}, img)
    noise = convert(Array{Float64, 2}, noise)
    # save("/home/alexander/Documents/alexander_leong/IPMPSDSolver.jl/data.jld", "img", img, "noise", noise)
end

function run_example()
    data = load("/home/alexander/Documents/alexander_leong/IPMPSDSolver.jl/data.jld")
    x_l, x_u, y_l, y_u = 49, 56, 49, 56
    img = data["img"]
    img = get_image_patch(img, x_l, y_l, x_u, y_u)
    noise = data["noise"]
    noise = get_image_patch(noise, x_l, y_l, x_u, y_u)
    cone_qp, sdp = denoise_image(img, noise)
    # data = load("/home/alexander/Documents/alexander_leong/IPMPSDSolver.jl/dataprep.jld")
    # cone_qp = data["prog"]
    # sdp = nothing
    solve(cone_qp)
    @info "Done!"
    # compute statistics between denoised image and ground truth
    # return denoised_img
end

# preprocess_data()
run_example()