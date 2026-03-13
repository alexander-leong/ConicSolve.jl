#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using DataStructures

mutable struct KKTIterate
    KKT_x
    s
    z
end

@enum ResultStatus begin
    NO_SOLUTION
    FEASIBLE_POINT
    INFEASIBLE_POINT
end

@enum TerminationStatus begin
    ALMOST_OPTIMAL
    DUAL_INFEASIBLE
    INFEASIBLE
    ITERATION_LIMIT
    NORM_LIMIT
    NUMERICAL_ERROR
    OBJECTIVE_LIMIT
    OPTIMIZE_NOT_CALLED
    OTHER_ERROR
    OTHER_LIMIT
    OPTIMAL
    SLOW_PROGRESS
    TIME_LIMIT
end

export TerminationStatus

@enum Device begin
    CPU
    GPU
end

export Device, CPU, GPU

@enum IterativeRefinementTriggerMode begin
    ITERATIVE_REFINEMENT_DEFAULT_TRIGGER_MODE
    ITERATIVE_REFINEMENT_PER_ITERATION
end

export IterativeRefinementTriggerMode, ITERATIVE_REFINEMENT_DEFAULT_TRIGGER_MODE, ITERATIVE_REFINEMENT_PER_ITERATION

function Base.parse(::Type{Device}, value)
    return value == "CPU" ? CPU : GPU
end

export parse

mutable struct SolverStatus
    best_iterate::Int32
    current_iteration::Int32
    current_residual_x::AbstractArray{Float64}
    current_residual_y::AbstractArray{Float64}
    current_residual_z::AbstractArray{Float64}
    kkt_iterate::CircularBuffer{KKTIterate}

    duality_gap::AbstractArray{Float64}
    dual_obj::AbstractArray{Float64}
    primal_obj::AbstractArray{Float64}
    residual_centrality::AbstractArray{Float64}
    residual_x::AbstractArray{Float64}
    residual_y::AbstractArray{Float64}
    residual_z::AbstractArray{Float64}
    step_size::AbstractArray{Float64}
    status_termination::Union{TerminationStatus, Nothing}

    function SolverStatus()
        status = new()
        status.best_iterate = 0
        status.current_iteration = 0
        status.current_residual_x = []
        status.current_residual_y = []
        status.current_residual_z = []
        buffer_size = 5
        status.kkt_iterate = CircularBuffer{KKTIterate}(buffer_size)
        status.duality_gap = []
        status.dual_obj = []
        status.primal_obj = []
        status.residual_centrality = []
        status.residual_x = []
        status.residual_y = []
        status.residual_z = []
        status.step_size = []
        status.status_termination = OPTIMIZE_NOT_CALLED
        return status
    end
end

function check_progress(status::SolverStatus, tol::Float64)
    if length(status.step_size) > 0 &&
    status.step_size[end] < tol
        if length(status.duality_gap) > 1 &&
        abs(status.duality_gap[end] - status.duality_gap[end-1]) < tol
            @info "Slow progress..."
            status.status_termination = SLOW_PROGRESS
        end
    end
end

function get_best_iterate(status::SolverStatus)
    rx_weight = 1
    ry_weight = 1
    rz_weight = 1
    gap_weight = 1
    solution_scores = []
    for (i, _) in enumerate(status.kkt_iterate)
        offset = length(status.kkt_iterate) - i + 1
        push!(solution_scores,
        rx_weight * status.residual_x[end-offset] +
        ry_weight * status.residual_y[end-offset] +
        rz_weight * status.residual_z[end-offset] +
        gap_weight * status.duality_gap[end-offset])
    end
    best_idx = argmin(solution_scores)
    i = status.current_iteration - best_idx
    @info "The primal/dual variables correspond to iterate $(best_idx)"
    return i, status.kkt_iterate[best_idx]
end

function print_table(data, header=false, n=100, pad=15)
    if header == true
        row = ""
        for (i, val) in enumerate(eachrow(data))
            if i == 1
                row *= lpad(val[:][1], 5, " ") * " |"
                continue
            end
            row *= lpad(val[:][1], pad, " ") * " |"
        end
        @info row
    end
    row = ""
    for (i, val) in enumerate(eachrow(data))
        if i == 1
            row *= lpad(val[:][end], 5, " ") * " |"
            continue
        end
        row *= lpad(round(val[:][end], sigdigits=6), pad, " ") * " |"
    end
    @info row
end

function log_iteration_status(status::SolverStatus, header=false, i=0, additional_data=[])
    data = ["Iter." status.current_iteration;
    "Dual Res." status.residual_x[end-i];
    "Primal Res." status.residual_y[end-i];
    "Cent. Res." status.residual_centrality[end-i];
    "Dual. Gap" status.duality_gap[end-i];
    "Primal Obj." status.primal_obj[end-i];
    "Dual Obj." status.dual_obj[end-i];
    "Step size" status.step_size[end-i];]
    if additional_data != []
        data = vcat(data, additional_data)
    end
    # TODO logging frequency and other options
    n = 100
    print_header = status.current_iteration % n == 1 || header
    print_table(data, print_header, n)
end

export log_iteration_status