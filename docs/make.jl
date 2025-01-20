using Documenter

push!(LOAD_PATH,"../src/")
using ConicSolve

makedocs(;
    authors="Alexander Leong <toshibaalexander@gmail.com>",
    sitename = "ConicSolve.jl",
    clean   = true, 
    doctest = false,
    modules = [ConicSolve],
    remotes = nothing,
    pages=[
        "Readme" => "index.md",
        "Tutorial" => "tutorial.md",
        "Examples" => ["examples/maximum_flow.md",
                       "examples/beamforming.md",
                       "examples/portfolio.md",
                       "examples/denoise.md"],
        "API Reference" => "functions.md"
    ],
    format = Documenter.HTML(
        mathengine = MathJax3(Dict(
            :loader => Dict("load" => ["[tex]/physics"]),
            :tex => Dict(
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
                "packages" => ["base", "ams", "autoload", "physics"],
            ),
        )),
        prettyurls = false,
        repolink="http://localhost:8000",
    )
)