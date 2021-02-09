using MLFlowLogger
using Documenter

DocMeta.setdocmeta!(MLFlowLogger, :DocTestSetup, :(using MLFlowLogger); recursive=true)

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
)

deploydocs(;
    repo="github.com/rejuvyesh/MLFlowLogger.jl",
)
