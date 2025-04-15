#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

import MathOptInterface as MOI

using ConicSolve

struct Optimizer <: MOI.AbstractOptimizer
    solver::Solver
end

function empty!(model::ModelLike)
end

function is_empty(model::ModelLike)
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    solver_name = "ConicSolve"
    return solver_name
end

function MOI.get(model::Optimizer, ::MOI.SolverVersion)
    version = "0.0.2"
    return version
end

function MOI.get(model::Optimizer, ::MOI.RawSolver)
    return model.solver
end

function MOI.get(model::Optimizer, ::MOI.Name)
end

function MOI.set(model::Optimizer, ::MOI.Name, v)
end

MOI.supports(::Optimizer, ::MOI.Name) = false

function MOI.get(model::Optimizer, ::MOI.Silent)
end

function MOI.set(model::Optimizer, ::MOI.Silent, v)
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, v)
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.get(model::Optimizer, ::MOI.ObjectiveLimit)
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveLimit, v)
end

MOI.supports(::Optimizer, ::MOI.ObjectiveLimit) = true

function MOI.get(model::Optimizer, ::MOI.SolutionLimit)
end

function MOI.set(model::Optimizer, ::MOI.SolutionLimit, v)
end

MOI.supports(::Optimizer, ::MOI.SolutionLimit) = true

function MOI.get(model::Optimizer, ::MOI.NodeLimit)
end

function MOI.set(model::Optimizer, ::MOI.NodeLimit, v)
end

MOI.supports(::Optimizer, ::MOI.NodeLimit) = true

function MOI.get(model::Optimizer, ::MOI.RawOptimizerAttribute)
end

function MOI.set(model::Optimizer, ::MOI.RawOptimizerAttribute, v)
end

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.get(model::Optimizer, ::MOI.NumberOfThreads)
end

function MOI.set(model::Optimizer, ::MOI.NumberOfThreads, v)
end

MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = true

function MOI.get(model::Optimizer, ::MOI.AbsoluteGapTolerance)
end

function MOI.set(model::Optimizer, ::MOI.AbsoluteGapTolerance, v)
end

MOI.supports(::Optimizer, ::MOI.AbsoluteGapTolerance) = true

function MOI.get(model::Optimizer, ::MOI.RelativeGapTolerance)
end

function MOI.set(model::Optimizer, ::MOI.RelativeGapTolerance, v)
end

MOI.supports(::Optimizer, ::MOI.RelativeGapTolerance) = true

function MOI.supports_constraint(::Optimizer, ::Vector{MOI.VariableIndex}, ::MOI.ConstraintIndex{MOI.VectorOfVariables, typeof(set)})
    return true
end

function optimize!(dest::Optimizer, src::ModelLike)
end

function MOI.get(model::Optimizer, ::MOI.DualStatus)
end

function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveValue)
end

function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
end

function MOI.get(model::Optimizer, ::MOI.VariablePrimal)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual)
end

function MOI.get(model::Optimizer, ::MOI.DualObjectiveValue)
end

function MOI.get(model::Optimizer, ::MOI.BarrierIterations)
end

function MOI.get(model::Optimizer, ::MOI.VariablePrimalStart)
end

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart, v)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDualStart)
end

function MOI.set(model::Optimizer, ::MOI.ConstraintDualStart, v)
end