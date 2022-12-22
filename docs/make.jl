using RecursiveTupleMath
using Documenter

DocMeta.setdocmeta!(RecursiveTupleMath, :DocTestSetup, :(using RecursiveTupleMath); recursive=true)

makedocs(;
    modules=[RecursiveTupleMath],
    authors="chriselrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/chriselrod/RecursiveTupleMath.jl/blob/{commit}{path}#{line}",
    sitename="RecursiveTupleMath.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/RecursiveTupleMath.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chriselrod/RecursiveTupleMath.jl",
    devbranch="main",
)
