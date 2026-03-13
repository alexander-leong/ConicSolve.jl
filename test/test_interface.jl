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

@testset "minimize l1 norm with vector c" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 3)

    A::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 4.]
    c = ones(6)

    define_program(program,
                minimize(l1(c, x)),
                A * x == b)
    program = build_program(program)
end

@testset "minimize l2 norm with vector c" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 3)

    A::Vector{Float64} = [1., 0., 0., 1., 0., 1.]
    b::Vector{Float64} = [1., 2., 4.]
    c = ones(4)

    define_program(program,
                minimize(l2(A, x)),
                A * x == b)
    program = build_program(program)
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
                lmi([A1, A2], x) in b)
    program = build_program(program)
end

@testset "expression less than zero wrt nonnegative orthant" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 6)

    A::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                A * x in NonNegativeOrthant(6))
    program = build_program(program)
end

@testset "expression less than zero wrt second order cone" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 6)

    A::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                A * x in SecondOrderCone(6))
    program = build_program(program)
end

@testset "expression less than zero wrt positive semidefinite cone" begin
    program = ConeQP()

    x = add_variable(program, NonNegativeOrthant(6), 6)

    A::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                A * x in PSDCone(6))
    program = build_program(program)
end

@testset "lmi in cone" begin
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
                lmi([A1, A2], x) ∈ b)
    program = build_program(program)
end

@testset "expression in nonnegative orthant" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 6)

    A::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                A * x ∈ NonNegativeOrthant(6))
    program = build_program(program)
end

@testset "expression in second order cone" begin
    program = ConeQP()

    x = add_variable(program, PSDCone(3), 6)

    A::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                A * x ∈ SecondOrderCone(6))
    program = build_program(program)
end

@testset "expression in positive semidefinite cone" begin
    program = ConeQP()

    x = add_variable(program, NonNegativeOrthant(6), 6)

    A::Matrix{Float64} = [1. 0. 0. 1. 0. 1.;
        0. 1. 0. 1. 0. 1.;
        0. 0. 1. 1. 0. 1.]
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                A * x ∈ PSDCone(6))
    program = build_program(program)
end

@testset "variable equal to vector" begin
    program = ConeQP()

    x = add_variable(program, NonNegativeOrthant(3), 3)
    
    b::Vector{Float64} = [1., 2., 3.]

    define_program(program,
                x == b)
    program = build_program(program)
end

@testset "index into variable" begin
end