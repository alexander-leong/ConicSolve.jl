include("./utils.jl")

using ConicSolve
using Logging

function julia_main()::Cint
    out_filepath, in_filepath = ARGS
    # println(out_filepath)
    # println(in_filepath)
    # in_filepath = "/home/alexander/Documents/alexander_leong/ConicSolve.jl/test/data/simple_problem.txt"
    solver = initialize_from_file(in_filepath)
    @info "OK!"
    run_solver(solver)
    # write_result_to_file(out_filepath, solver)
    return 0
end