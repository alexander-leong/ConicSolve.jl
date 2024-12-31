using Documenter

push!(LOAD_PATH,"../src/")
using ConicSolve

makedocs(;
    authors="Alexander Leong <toshibaalexander@gmail.com>",
    sitename = "ConicSolve.jl", 
    doctest = false,
    modules = [ConicSolve],
    remotes = nothing,
    pages=[
        "Readme" => "index.md",
        "Tutorial" => "tutorial.md",
        "Functions" => "functions.md"
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