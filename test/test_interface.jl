using ConicSolve
using Test

@testset "matrix multiply variable" begin
    program = ConeQP()

    x = add_variable(program, NonNegativeOrthant(4), 4)

    A::Matrix{Float64} = [1. 0. 0. 1.;
        0. 1. 0. 1.;
        0. 0. 1. 1.]
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                A * x == b)
    program = build_program(program)
end

@testset "vector multiply variable" begin
    program = ConeQP()

    x = add_variable(program, NonNegativeOrthant(4), 4)

    A::Vector{Float64} = [1., 0., 0., 1.]
    b::Vector{Float64} = [1.]

    define_program(program,
                A * x == b)
    program = build_program(program)
end

@testset "scalar multiply variable" begin
    program = ConeQP()

    x = add_variable(program, NonNegativeOrthant(4), 4)

    b::Vector{Float64} = [1., 0., 0., 1.]

    define_program(program,
                2. * x == b)
    program = build_program(program)
end

@testset "expression plus expression" begin
    program = ConeQP()

    x1 = add_variable(program, NonNegativeOrthant(6), 6)
    x2 = add_variable(program, PSDCone(3), 3)

    A1::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    A2::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 1. 1. 0. 1.;
        1. 0. 0. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 4., 3., 1., 2.]

    define_program(program,
                (A1 * x1) + (A2 * x2) == b)
    program = build_program(program)
end

@testset "expression minus expression" begin
    program = ConeQP()

    x1 = add_variable(program, NonNegativeOrthant(6), 6)
    x2 = add_variable(program, PSDCone(3), 3)

    A1::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    A2::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 1. 1. 0. 1.;
        1. 0. 0. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 4., 3., 1., 2.]

    define_program(program,
                (A1 * x1) - (A2 * x2) == b)
    program = build_program(program)
end

@testset "l1 norm of vector wrt cone" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 3)

    A::Vector{Float64} = [1., 0., 0., 1., 0., 1.]
    b::Vector{Float64} = [1., 2., 4.]

    define_program(program,
                l1(A, x) == b)
    program = build_program(program)
end

@testset "l1 norm of matrix wrt cone" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 3)

    A::Matrix{Float64} = [1. 0. 0. 1.;
        0. 1. 0. 1.;
        0. 0. 1. 1.]
    b::Vector{Float64} = [1., 2., 4.]

    define_program(program,
                l1(A, x) == b)
    program = build_program(program)
end

@testset "l2 norm of vector wrt cone" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 3)

    A::Vector{Float64} = [1., 0., 0., 1., 0., 1.]
    b::Vector{Float64} = [1., 2., 4.]

    define_program(program,
                l2(A, x) == b)
    program = build_program(program)
end

@testset "l2 norm of matrix wrt cone" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 3)

    A::Matrix{Float64} = [1. 0. 0. 1.;
        0. 1. 0. 1.;
        0. 0. 1. 1.]
    b::Vector{Float64} = [1., 2., 4.]

    define_program(program,
                l2(A, x) == b)
    program = build_program(program)
end

@testset "lmi in vectorized form wrt cone" begin
end

@testset "lmi in matrix form wrt cone" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 3)

    A1::Matrix{Float64} = [1. 0. 1.;
        0. 1. 0.;
        1. 0. 1.]
    A2::Matrix{Float64} = [5. 1. 1.;
        1. 4. 0.;
        1. 0. 3.]
    b::Matrix{Float64} = [1. 0. 1.;
        0. 8. 0.;
        1. 0. 1.]

    define_program(program,
                lmi([A1, A2], x) <= b)
    program = build_program(program)
end

@testset "variable less than vector wrt nonnegative orthant" begin
end

@testset "expression less than vector wrt second order cone" begin
end

@testset "variable less than matrix wrt positive semidefinite cone" begin
end

@testset "expression less than matrix wrt positive semidefinite cone" begin
end

@testset "expression equal to vector" begin
end

@testset "intersect two cones" begin
end

@testset "add cone to set intersection of cones" begin
end

@testset "index into variable" begin
end