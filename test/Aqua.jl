using Aqua
using ConicSolve
using Test

@testset "Aqua.jl" begin
  Aqua.test_all(
    ConicSolve;
    piracies=false,
  )
end