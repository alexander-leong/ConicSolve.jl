using Documenter

push!(LOAD_PATH,"../src/")
using IPMPSDSolver

makedocs(;
    authors="Alexander Leong <toshibaalexander@gmail.com>",
    sitename = "ConicSolve.jl", 
    doctest = false,
    remotes = nothing,
    pages=[
        "Readme" => "index.md",
        "Tutorial" => "tutorial.md",
        "Functions" => "functions.md"
    ],
    format = Documenter.HTML(
        prettyurls = false,
        repolink="http://localhost:8000",
    )
)