using ConicSolve
using JLD
using LinearAlgebra
using Test

include("../src/arrayutils.jl")

@testset "get_diag_dominant_multiplier" begin
    A = [1 2 3 4 5; 2 6 7 8 9; 3 7 -10 -11 12; 4 8 -11 13 14; 5 9 12 14 15]
    s = minimum(eigen(A).values)
    println(eigen(A + abs(s)*I).values)
    A = -A
    s = minimum(eigen(A).values)
    println(eigen(A + abs(s)*I).values)
end

@testset "dropzero_rows" begin
    X = Matrix{Float64}(I, 6, 6)
    X[2, 2] = 0
    X[5, 5] = 0
    X[6, 6] = 0
    new_X, inds = dropzero_rows(X)
    @test sum(new_X) == 3
    @test size(new_X) == (3, 6)
    @test new_X[1, 1] == new_X[2, 3] == new_X[3, 4] == 1
    @test inds == [1, 3, 4]
end

@testset "flatten_2d" begin
    dim = (8, 8)
    A_idx = CartesianIndices(dim)
    A_idx = A_idx[6:7, 3:7]
    mask_img_idx = flatten_2d(A_idx, dim[2])
    @test mask_img_idx == [43 44 45 46 47; 51 52 53 54 55]
end

@testset "get_img_idx_in_A" begin
    filepath = ""
    data = load(filepath)
    x_l, x_u, y_l, y_u = 1, 8, 1, 8
    img = data["img"]
    img = img[x_l:x_u, y_l:y_u]
    dim = (16, 16)
    A_idx = CartesianIndices(dim)
    idx = get_img_idx_in_A(A_idx, img)
    @test idx == CartesianIndices((1:8, 9:16))[:]
end

@testset "get_img_transposed_idx_in_A" begin
    filepath = ""
    data = load(filepath)
    x_l, x_u, y_l, y_u = 1, 8, 1, 8
    img = data["img"]
    img = img[x_l:x_u, y_l:y_u]
    dim = (16, 16)
    A_idx = CartesianIndices(dim)
    idx = get_img_transposed_idx_in_A(A_idx, img)
    @test idx == CartesianIndices((9:16, 1:8))[:]
end

@testset "get_mask" begin
    filepath = ""
    data = load(filepath)
    x_l, x_u, y_l, y_u = 1, 4, 1, 4
    noise = data["noise"]
    noise = noise[x_l:x_u, y_l:y_u]
    signal = get_mask(x -> x == 0, noise)
    @test signal == Bool[1 1 1 1; 1 1 1 1; 0 0 1 1; 1 1 1 1]
end

@testset "ohe_from_2d_mask" begin
    filepath = ""
    data = load(filepath)
    x_l, x_u, y_l, y_u = 1, 2, 1, 2
    img = data["img"]
    img = img[x_l:x_u, y_l:y_u]
    noise = data["noise"]
    noise = noise[x_l:x_u, y_l:y_u]
    dim = (4, 4)
    X_size = (dim[1] + dim[2])^2
    A_idx = CartesianIndices(dim)
    A = zeros((X_size, X_size))
    nCols = 4
    idx = get_img_idx_in_A(A_idx, img)
    offsetX = 0
    offsetY = size(A_idx)[2] - size(img)[2]
    ohe_from_2d_mask(A, idx, noise, nCols, offsetX, offsetY)
    @test sum(A) == 4
    @test unique(A) == [0, 1]
    @test A[3, 3] == A[4, 4] == A[7, 7] == A[8, 8] == 1
end

@testset "mat" begin
    u = [11, sqrt(2)*21, sqrt(2)*31, 22, sqrt(2)*23, 33]
    U = mat(u)
    @test isapprox(U, [11 21 31;
    21 22 23;
    31 23 33])
end

@testset "svec" begin
    U = [11 21 31;
    21 22 23;
    31 23 33]
    u = svec(U)
    @test isapprox(u, [11, sqrt(2)*21, sqrt(2)*31, 22, sqrt(2)*23, 33])
end