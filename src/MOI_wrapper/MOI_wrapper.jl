#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import MathOptInterface as MOI

mutable struct Optimizer <: MOI.AbstractOptimizer
    solver
    status
    
    function Optimizer()
        return new(
            missing,
            MOI.OPTIMIZE_NOT_CALLED
        )
    end
end

export Optimizer
function empty!(model::MOI.ModelLike)
end

function is_empty(model::MOI.ModelLike)
end

function MOI.empty!(opt::Optimizer)
    model.solver = missing
    return
end

function MOI.is_empty(opt::Optimizer)
    return ismissing(model.solver) && model.status == MOI.OPTIMIZE_NOT_CALLED
end

function MOI.get(opt::Optimizer, ::MOI.SolverName)
    solver_name = "ConicSolve"
    return solver_name
end

function MOI.get(opt::Optimizer, ::MOI.SolverVersion)
    version = "0.0.2"
    return version
end

function MOI.get(opt::Optimizer, ::MOI.RawSolver)
    return model.solver
end

function MOI.get(opt::Optimizer, ::MOI.Name)
end

function MOI.set(opt::Optimizer, ::MOI.Name, v)
end

MOI.supports(::Optimizer, ::MOI.Name) = false

function MOI.get(opt::Optimizer, ::MOI.Silent)
    return occursin("ConicSolve", ENV["JULIA_DEBUG"])
end

function MOI.set(opt::Optimizer, ::MOI.Silent, v)
    if v
        ENV["JULIA_DEBUG"] = ""
    else
        ENV["JULIA_DEBUG"] = "ConicSolve"
    end
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.get(opt::Optimizer, ::MOI.TimeLimitSec)
    return model.solve_time
end

function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, v)
    model.solve_time = v
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.get(opt::Optimizer, ::MOI.ObjectiveLimit)
    return model.limit_obj
end

function MOI.set(opt::Optimizer, ::MOI.ObjectiveLimit, v)
    model.limit_obj = v
end

MOI.supports(::Optimizer, ::MOI.ObjectiveLimit) = true

function MOI.get(opt::Optimizer, ::MOI.SolutionLimit)
end

function MOI.set(opt::Optimizer, ::MOI.SolutionLimit, v)
end

MOI.supports(::Optimizer, ::MOI.SolutionLimit) = false

function MOI.get(opt::Optimizer, ::MOI.NodeLimit)
end

function MOI.set(opt::Optimizer, ::MOI.NodeLimit, v)
end

MOI.supports(::Optimizer, ::MOI.NodeLimit) = false

function MOI.get(opt::Optimizer, ::MOI.RawOptimizerAttribute)
end

function MOI.set(opt::Optimizer, ::MOI.RawOptimizerAttribute, v)
end

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = false

function MOI.get(opt::Optimizer, ::MOI.NumberOfThreads)
    numThreads = 1
    return numThreads
end

function MOI.set(opt::Optimizer, ::MOI.NumberOfThreads, v)
end

MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = true

function MOI.get(opt::Optimizer, ::MOI.AbsoluteGapTolerance)
    return model.tol_gap_abs
end

function MOI.set(opt::Optimizer, ::MOI.AbsoluteGapTolerance, v)
    model.tol_gap_abs = v
end

MOI.supports(::Optimizer, ::MOI.AbsoluteGapTolerance) = true

function MOI.get(opt::Optimizer, ::MOI.RelativeGapTolerance)
    return model.tol_gap_rel
end

function MOI.set(opt::Optimizer, ::MOI.RelativeGapTolerance, v)
    model.tol_gap_rel = v
end

MOI.supports(::Optimizer, ::MOI.RelativeGapTolerance) = true

function MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{MOI.Nonnegatives})
    return true
end

function MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{MOI.SecondOrderCone})
    return true
end

function MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{MOI.PositiveSemidefiniteConeTriangle})
    return true
end

function MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorAffineFunction{Float64}}, ::Type{MOI.Zeros})
    return true
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}})
    return true
end

MOI.Utilities.@product_of_sets(Cones,
                               MOI.Zeros,
                               MOI.Nonnegatives,
                               MOI.SecondOrderCone,
                               MOI.PositiveSemidefiniteConeTriangle)

const CacheModel = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing
        },
        Vector{Float64},
        Cones{Float64}
    }
}

function add_model_info(model::CacheModel, set)
    ext = model.ext
    push!(ext[:cones], typeof(set))
    push!(ext[:indices], MOI.get(model, MOI.NumberOfVariables()))
    x = MOI.add_variables(model, MOI.dimension(set))
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables, typeof(set)}(x[1].value)
    return x, ci
end

function MOI.add_constrained_variables(model::CacheModel, set::MOI.Nonnegatives)
    x, ci = add_model_info(model, set)
    return x, ci
end

function MOI.add_constrained_variables(model::CacheModel, set::MOI.SecondOrderCone)
    x, ci = add_model_info(model, set)
    return x, ci
end

function MOI.add_constrained_variables(model::CacheModel, set::MOI.PositiveSemidefiniteConeTriangle)
    x, ci = add_model_info(model, set)
    return x, ci
end

function resolve_constraints(index_map, src, num_rows)
    constraints = []
    n = []
    for con in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables, MOI.Nonnegatives}())
        idx = index_map[con].value
        push!(constraints, (MOI.Nonnegatives, idx))
        push!(n, idx)
    end
    for con in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables, MOI.SecondOrderCone}())
        idx = index_map[con].value
        push!(constraints, (MOI.SecondOrderCone, idx))
        push!(n, idx)
    end
    for con in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle}())
        idx = index_map[con].value
        push!(constraints, (MOI.PositiveSemidefiniteConeTriangle, idx))
        push!(n, idx)
    end
    sort!(constraints, by = x -> x[end])
    push!(n, num_rows + 1)
    n = diff(n)
    return constraints, n
end

function initialize_solver!(A, b, c, dest, src, index_map, constraints, m, n)
    # Extract problem input matrices, vectors from the JuMP model
    F, S = MOI.VectorAffineFunction{Float64}, MOI.Zeros
    conInds = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())[end]
    numEqualityConstraints = index_map[conInds].value - 1
    G = A[numEqualityConstraints+1:end, 1:m]
    G = -collect(G)
    A = A[1:numEqualityConstraints, 1:m]
    A = collect(A)
    P = zeros((m, m))
    h = -b[numEqualityConstraints+1:end]
    b = b[1:numEqualityConstraints]

    # Cones need to be pushed in the order that is defined in the matrix G
    cones::Vector{Cone} = []
    for (i, constraint) in enumerate(constraints)
        constraint_type = first(constraint)
        if constraint_type == MOI.Nonnegatives
            push!(cones, NonNegativeOrthant(n[i]))
        elseif constraint_type == MOI.SecondOrderCone
            push!(cones, SecondOrderCone(n[i]))
        elseif constraint_type == MOI.PositiveSemidefiniteConeTriangle
            push!(cones, PositiveSemidefiniteCone(n[i]))
        end
    end

    # Create an instance of the ConicSolve.jl solver with the input matrices
    # and conic constraint types, Float64 only supported for now.
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    dest.solver = Solver(cone_qp, "qrchol")
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = CacheModel()
    cache.ext = Dict(:cones => [], :indices => [])
    index_map = MOI.copy_to(cache, src)
    A = convert(
            SparseArrays.SparseMatrixCSC{Float64, Int},
            cache.constraints.coefficients)
    b = -cache.constraints.constants
    num_rows = MOI.get(src, MOI.NumberOfVariables())
    constraints, n = resolve_constraints(index_map, src, num_rows)
    sense = ifelse(cache.objective.sense == MOI.MAX_SENSE, -1, 1)
    F = MOI.ScalarAffineFunction{Float64}
    obj = MOI.get(src, MOI.ObjectiveFunction{F}())
    m = first(cache.ext[:indices])
    c = zeros(m)
    for term in obj.terms
        c[term.variable.value] += sense * term.coefficient
    end
    initialize_solver!(A, b, c, dest, src, index_map, constraints, m, n)
    run_solver(dest.solver)
    # TODO: record the solution
    return index_map, false
end

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    solver = opt.solver
    return solver.status_dual
end

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    solver = opt.solver
    return solver.status_primal
end

function MOI.get(opt::Optimizer, ::MOI.RawStatusString)
    solverStatus = model.status
    return solverStatus.status_termination
end

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    return 1
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    solverStatus = model.status
    return solverStatus.status_termination
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    return model.obj_primal_value
end

function MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)
    return model.solve_time
end

function MOI.get(opt::Optimizer, ::MOI.VariablePrimal)
    program = model.program
    result = get_variable_primal(program)
    return result
end

function MOI.get(opt::Optimizer, ::MOI.ConstraintDual)
    program = model.program
    result = get_constraint_dual(program)
    return result
end

function MOI.get(opt::Optimizer, ::MOI.DualObjectiveValue)
    # TODO: Implement functionality
end

function MOI.get(opt::Optimizer, ::MOI.BarrierIterations)
    solverStatus = model.status
    return solverStatus.current_iteration
end

function MOI.get(opt::Optimizer, ::MOI.VariablePrimalStart)
    # TODO: Implement functionality
end

function MOI.set(opt::Optimizer, ::MOI.VariablePrimalStart, v)
    # TODO: Implement functionality
end

function MOI.get(opt::Optimizer, ::MOI.ConstraintDualStart)
    # TODO: Implement functionality
end

function MOI.set(opt::Optimizer, ::MOI.ConstraintDualStart, v)
    # TODO: Implement functionality
end