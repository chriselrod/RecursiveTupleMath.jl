# RecursiveTupleMath

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/RecursiveTupleMath.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriselrod.github.io/RecursiveTupleMath.jl/dev/)
[![Build Status](https://github.com/chriselrod/RecursiveTupleMath.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chriselrod/RecursiveTupleMath.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/chriselrod/RecursiveTupleMath.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/chriselrod/RecursiveTupleMath.jl)

What is the motivation of this repo?
1. The language server doesn't understand defining methods with macros like `@eval` or `ForwardDiff.@define_binary_dual_op`. Defining methods within the module you're working on will result in them all being marked as undefined references, but if a dependency defines them, the LSP will be able see them. Therefore, it's nice to silo off all macro-defined methods into a separate repo.
2. Sometimes it's nice to work with and do math with tuples.
