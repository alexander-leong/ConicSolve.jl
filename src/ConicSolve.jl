#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

module ConicSolve

include("./cones/cone.jl")
include("./cones/nonneg.jl")
include("./cones/psdcone.jl")
include("./cones/soc.jl")
include("./models/sdp.jl")
include("./models/sos.jl")
include("solver.jl")
include("MOI_wrapper/MOI_wrapper.jl")
include("utils.jl")

Base.@ccallable function julia_main()::Cint
    out_filepath, in_filepath = ARGS
    try
        solver = initialize_from_file(in_filepath)
        run_solver(solver)
        write_result_to_file(out_filepath, solver)
    catch err
        @error err
    end
    return 0
end

end