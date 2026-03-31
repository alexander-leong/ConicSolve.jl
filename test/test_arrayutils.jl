using ConicSolve
using LinearAlgebra
using Test

include("../src/arrayutils.jl")

@testset "lower_triangular_from_2d_idx" begin
end

@testset "ohe_from_2d_mask" begin
    x_l, x_u, y_l, y_u = 1, 2, 1, 2
    n = 2
    mask = [1 1; 0 1]
    mask = mask[x_l:x_u, y_l:y_u]
    dim = (4, 4)
    X_size = (dim[1] + dim[2])^2
    A_idx = CartesianIndices(dim)
    A = zeros((X_size, X_size))
    nCols = 4
    idx = CartesianIndices((x_l:x_u, y_l:y_u))[:]
    offsetX = 0
    offsetY = 0
    ohe_from_2d_mask(A, idx, mask, nCols, offsetX, offsetY)
    @test unique(A) == [0, 1]
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
    img = rand(8, 8)
    dim = (16, 16)
    A_idx = CartesianIndices(dim)
    idx = get_img_idx_in_A(A_idx, img)
    @test idx == CartesianIndices((1:8, 9:16))[:]
end

@testset "get_img_transposed_idx_in_A" begin
    img = rand(8, 8)
    dim = (16, 16)
    A_idx = CartesianIndices(dim)
    idx = get_img_transposed_idx_in_A(A_idx, img)
    @test idx == CartesianIndices((9:16, 1:8))[:]
end

@testset "get_mask" begin
    noise = [0 0 0 0; 0 0 0 0; 1 1 0 0; 0 0 0 0]
    signal = get_mask(x -> x == 0, noise)
    @test signal == Bool[1 1 1 1; 1 1 1 1; 0 0 1 1; 1 1 1 1]
end

@testset "mat" begin
    u = [11, sqrt(2)*21, sqrt(2)*31, 22, sqrt(2)*23, 33]
    U = mat(u)
    @test isapprox(U, [11. 21. 31.;
    21. 22. 23.;
    31. 23. 33.])
end

@testset "svec" begin
    U = [11. 21. 31.;
    21. 22. 23.;
    31. 23. 33.]
    u = svec(U)
    @test isapprox(u, [11, sqrt(2)*21, sqrt(2)*31, 22, sqrt(2)*23, 33])
end