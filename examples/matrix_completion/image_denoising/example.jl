#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

""" Matrix completion example for image denoising
via Nuclear-norm minimization and SDP embedding lemma.
see Matrix Rank Minimization with Applications, Fazel (2002)
"""

# include("./imageutils.jl")

using ConicSolve

function run_example(data, mask)
    program = ConeQP()
    A = FixedValue
    b = (data, mask)
    x = add_variable(program, NuclearNorm, size(data))
    
    define_program(program,
                minimize(nuclear_norm(x)),
                A * x == b,
                x ∈ NonNegativeOrthant())
    program = build_program(program)
    
    solver = Solver(program)
    # solver.max_iterations = 10
    # status = run_solver(solver)
    # x = get_solution(solver)
    # return solver, status
    return nothing, nothing
end

# data, mask = preprocess_data()
data = [1. 2. 3. 4.; 5. 6. 7. 8.; 9. 10. 11. 12.]
mask = Bool.([0 0 0 0; 0 0 1 0; 1 0 0 1])
solver, status = run_example(data, mask)