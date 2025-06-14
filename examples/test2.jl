using ConicSolve
using CairoMakie
using LinearAlgebra

function run_solver_on_example()
    A = [] # empty
    # first line - (0, 10), (10, 15)
    # y - (1/2)*x <= 10
    # second line - (0, 10), (2, 0)
    # y + 5*x >= 10
    # third line - (2, 0), (10, 15)
    # y - (15/8)*x >= -3.75
    G = [2 1; -1/5 -1; -8/15 -1]
    P = [1/3 0; 0 1/4]
    b = [] # empty
    c = zeros(Float64, 2)
    h = [10, 10, -3.75] # h - Gx >= 0
    cones::Vector{Cone} = []
    push!(cones, NonNegativeOrthant(3))
    cone_qp = ConeQP{Float64, Float64, Float64}(nothing, G, P, nothing, c, h, cones)
    kktsolve = "qrchol"
    solver = Solver(cone_qp, kktsolve)
    results = []
    function cb_before_iteration(solver::Solver)
        if solver.current_iteration == 1
            push!(results, get_solution(solver))
        end
    end
    function cb_after_iteration(solver::Solver)
        push!(results, get_solution(solver))
    end
    solver.cb_after_iteration = cb_after_iteration
    solver.cb_before_iteration = cb_before_iteration
    solver.max_iterations = 10
    run_solver(solver, false, solver)
    status = solver.status

    f = Figure()
    Axis(f[1, 1])

    xs = LinRange(-5, 10, 100)
    ys = LinRange(-5, 15, 100)
    zs = [(1/3*x)^2 + (1/4*y)^2 for x in xs, y in ys]

    contour!(xs, ys, zs, color=:grey)

    xl = [0, 2]
    yl = [10, 0]
    lines!(xl, yl; color=:black)
    xl = [0, 10]
    yl = [10, 15]
    lines!(xl, yl; color=:black)
    xl = [2, 10]
    yl = [0, 15]
    lines!(xl, yl; color=:black)

    # pts::Vector{Tuple{Float64, Float64}} = [
    #     (7.136551122140249, -4.84668504065929),
    #     (-0.24818355788293456, 4.67908971595938),
    #     (0.06393643520464676, 3.6641341393286795),
    #     (1.046060296990675, 3.1816608575137106),
    #     (1.230581541349875, 3.0936519816497694),
    #     (1.2324584977830666, 3.092606961346143),
    #     (1.2362133169388307, 3.090685819256488)]
    pts::Vector{Tuple{Float64, Float64}} = []
    for pt in results
        push!(pts, (pt[1], pt[2]))
    end
    lines!(pts, linestyle=:solid, color=:blue)
    scatter!(pts, color=:blue)
    display(f)
    return results, status
end

include("total_variation_reconstruction/example.jl")
function run_solver_on_any_example()
    y, cone_qp = run_example()
    kktsolve = "qrchol"
    solver = Solver(cone_qp, kktsolve)
    results = []
    function cb_before_iteration(solver::Solver)
        if solver.current_iteration == 1
            push!(results, get_solution(solver))
        end
    end
    function cb_after_iteration(solver::Solver)
        push!(results, get_solution(solver))
    end
    solver.cb_after_iteration = cb_after_iteration
    solver.cb_before_iteration = cb_before_iteration
    solver.max_iterations = 10
    run_solver(solver, false, solver)
    status = solver.status
    return y, status, results
end

# function render_example()
y, status, results = run_solver_on_any_example()
f = Figure()
x = 1:1:80
ax = Axis(f[1, 1], title="Input Signal over time", limits=(1, 80, -3, 2))
lines!(ax, x, y)
ax = Axis(f[2, 1], title="Reconstructed Signal over time", limits=(1, 80, -3, 2))
y = results[end][1:80]
lines!(ax, x, y)
display(f)

# g = Figure()
# g_ax_1 = Axis(g[1, 1], title="Residuals x over Iterations", xlabel="Iteration #", ylabel="||Residual x||_2")
# g_ax_2 = Axis(g[1, 2], title="Residuals y over Iterations", xlabel="Iteration #", ylabel="||Residual y||_2")
# g_ax_3 = Axis(g[2, 1], title="Residuals z over Iterations", xlabel="Iteration #", ylabel="||Residual z||_2")
# g_ax_4 = Axis(g[2, 2], title="Duality gap over Iterations", xlabel="Iteration #", ylabel="Duality Gap")
# x::Vector{Int} = []
# y_res_x_2_norm::Vector{Float64} = status.residual_x
# y_res_y_2_norm::Vector{Float64} = status.residual_y
# y_res_z_2_norm::Vector{Float64} = status.residual_z
# y_duality_gap::Vector{Float64} = status.duality_gap
# x = 1:1:length(y_duality_gap)
# lines!(g_ax_1, x, y_res_x_2_norm)
# lines!(g_ax_2, x, y_res_y_2_norm)
# lines!(g_ax_3, x, y_res_z_2_norm)
# lines!(g_ax_4, x, y_duality_gap)
# display(g)

    # return f
# end

# render_example()

# export run_example
# export render_example