using ConicSolve
using DynamicPolynomials
using Test

@testset "polynomial substitute" begin
    @polyvar t
    @polyvar q[1:4]

    p = q[4]*t^4 + q[3]*t^3 + q[2]*t^2 + q[1]*t
    p_sub = polynomial_substitute(p, t=>2.)
    println(p_sub)
end