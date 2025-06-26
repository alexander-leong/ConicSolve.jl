#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using CSV

function read_vector(line)
    return parse.(Float64, split(line, r"(,\s*)"))
end

function read_matrix(lines)
    f = CSV.File(IOBuffer(lines); header=false)
    return f|>CSV.Tables.matrix
end

function initialize_from_file(filepath)
    T = nothing
    cone_qp = ConeQP()
    cones::Vector{Cone} = []
    solver = nothing
    obj = nothing
    symbol = nothing
    words = readlines(filepath)
    value = ""
    for (i, word) in enumerate(words)
        if word == ""
            word = "\n"
        end
        if word == "ConeQP"
            obj = cone_qp
            continue
        end
        if word == "Solver"
            T = Solver
            cone_qp.cones = obj
            solver = Solver(cone_qp)
            obj = solver
            continue
        end
        if word in ["A", "G", "P"]
            T = Matrix
            symbol = Meta.parse(word)
            value = ""
            continue
        end
        if word in ["b", "c", "h"]
            T = Vector
            symbol = Meta.parse(word)
            value = ""
            continue
        end
        if word == "cones"
            cone_qp = obj
            T = Vector{Cone}
            obj = cones
            continue
        end
        if T == Vector{Cone}
            if occursin("NonNegativeOrthant", word)
                p = parse(Int, split(word, r"(,\s*)")[end])
                push!(obj, NonNegativeOrthant(p))
                continue
            end
            if occursin("SecondOrderCone", word)
                p = parse(Int, split(word, r"(,\s*)")[end])
                push!(obj, SecondOrderCone(p))
                continue
            end
            if occursin("PSDCone", word)
                p = parse(Int, split(word, r"(,\s*)")[end])
                push!(obj, PSDCone(p))
                continue
            end
        end
        if T == Solver
            symbol, value = split(word, r"(,\s*)")
            symbol = Meta.parse(symbol)
            value = parse(typeof(getproperty(obj, symbol)), value)
            setproperty!(obj, symbol, value)
            solver = obj
            continue
        end
        if i < length(words) && word != "\n"
            # still reading in value
            value = value * word * "\n"
        end
        if words[i+1] == ""
            if T == Vector{Cone}
                cones = obj
            else
                if T == Vector
                    value = read_vector(value)
                    setproperty!(obj, symbol, value)
                end
                if T == Matrix
                    value = read_matrix(value)
                    setproperty!(obj, symbol, value)
                end
            end
        end
    end
    return solver
end

function write_result_to_file(filepath, solver)
    fp = open(filepath, "w")

    program = solver.program
    status = get_solver_status(solver)

    write(fp, "PRIMAL STATUS: " * string(solver.status_primal) * "\n")
    write(fp, "DUAL STATUS: " * string(solver.status_dual) * "\n")
    write(fp, "TERMINATION STATUS: " * string(status.status_termination) * "\n")
    
    write(fp, "x = " * string(program.KKT_x[program.inds_c]) * "\n")
    if program.A != undef && !isnothing(program.A)
        write(fp, "y = " * string(program.KKT_x[program.inds_b]) * "\n")
    end
    write(fp, "z = " * string(program.KKT_x[program.inds_h]) * "\n")
    write(fp, "s = " * string(program.s) * "\n")
    
    close(fp)
end