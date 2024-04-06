using MLFlowLogger
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(MLFlowLogger, :DocTestSetup, :(using MLFlowLogger); recursive=true)
links = InterLinks(
    "Julia" => (
        "https://docs.julialang.org/en/v1/",
        joinpath(dirname(dirname(pathof(DocumenterInterLinks))), "docs/src/inventories/Julia.toml")        
    ),
)
makedocs(;
    modules=[MLFlowLogger],
    authors="rejuvyesh <mail@rejuvyesh.com> and contributors",
    repo="https://github.com/rejuvyesh/MLFlowLogger.jl/blob/{commit}{path}#{line}",
    sitename="MLFlowLogger.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rejuvyesh.github.io/MLFlowLogger.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    plugins=[links],
)

deploydocs(;
    repo="github.com/rejuvyesh/MLFlowLogger.jl",
)
