#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ColorTypes
using ImageFiltering
# using JLD
using TestImages

function load_image()
    img = testimage("cameraman")
    return img
end

function add_noise(img, noise_level)
    n = noise_level
    # apply salt-pepper noise to image
    # recipe copied from ImageFiltering.jl example
    noisy_img = map(img) do p
        sp = rand()
        if sp < n/2
            eltype(img)(gamutmin(eltype(img))...)
        elseif sp < n
            eltype(img)(gamutmax(eltype(img))...)
        else
            p
        end
    end
    return noisy_img
end

function detect_noise(img, threshold)
    # get noisy pixels by applying laplacian filter
    # 0 for no noise, 1 for noise
    imgl = imfilter(img, Kernel.Laplacian())
    noise = map(x -> x >= threshold, imgl)
    return noise
end

# function preprocess_data()
#     img = load_image()
#     noisy_img = add_noise(img, 0.2)
#     threshold = 0.99
#     noise = detect_noise(noisy_img, threshold)

#     img = convert(Array{Float64, 2}, img)
#     noise = convert(Array{Float64, 2}, noise)
    
#     data = load("/home/alexander/Documents/alexander_leong/ConicSolve.jl/data.jld")
#     x_l, x_u, y_l, y_u = 107, 154, 225, 272
#     img = data["img"]
#     noise = data["noise"]
#     img = get_block_matrix(img, x_l, y_l, x_u, y_u)
#     noise = get_block_matrix(noise, x_l, y_l, x_u, y_u)
#     return img, noise
# end
