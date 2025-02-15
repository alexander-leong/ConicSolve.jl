# using ConicSolve

include("beamforming/example.jl")
beamforming_status = run_example()
include("matrix_completion/image_denoising/example.jl")
denoising_status = run_example()
include("maximum_flow/example.jl")
max_flow_status = run_example()
include("trajectory_optimization/example.jl")
traj_opt_status = run_example()

status = [beamforming_status, denoising_status, max_flow_status, traj_opt_status]
series_labels = ["beamforming", "matrix_completion", "maximum flow", "trajectory optimization"]

x_ranges = [range(1, x.current_iteration) for x in status]
gap = [x.duality_gap for x in status]

using CairoMakie

# Create a figure and axis
fig = Figure(resolution=(600, 400))
ax = Axis(fig[1, 1], title="Duality Gap", xlabel="Iteration #", ylabel="Duality Gap")

# Define different-length datasets
x_data = x_ranges
y_data = gap

# Define colors for each line
colors = [:red, :blue, :green, :purple]

# Plot each series separately
for (i, (x, y)) in enumerate(zip(x_data, y_data))
    lines!(ax, x, y, color=colors[i], label=series_labels[i])
end

# Add a legend
axislegend()

# Save the figure (optional)
save("duality_gap.png", fig)

# Display the figure
display(fig)
