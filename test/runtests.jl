using Base: Forward
using RecursiveTupleMath, StaticArrays, ForwardDiff
using Test

foo(x, y, z) = sum(min.(z .+ x, (x .- y)) ./ max.(z .- y, (x .* y)))
bar(x, y, z) = sum(bdiv(bmin(badd(z, x), bsub(x, y)), bmax(bsub(z, y), bmul(x, y))))

@testset "RecursiveTupleMath.jl" begin
  x = rand()
  xv = @SVector rand(3)
  for y in (rand(), @SVector(rand(3))), z in (rand(), @SVector(rand(3)))

    @test ForwardDiff.derivative(x -> foo(x, y, z), x) ≈
          ForwardDiff.derivative(x -> bar(x, y, z), x)
    @test ForwardDiff.derivative(x -> foo(y, x, z), x) ≈
          ForwardDiff.derivative(x -> bar(y, x, z), x)
    @test ForwardDiff.derivative(x -> foo(y, z, x), x) ≈
          ForwardDiff.derivative(x -> bar(y, z, x), x)

    @test ForwardDiff.gradient(x -> foo(x, y, z), xv) ≈
          ForwardDiff.gradient(x -> bar(x, y, z), xv)
    @test ForwardDiff.gradient(x -> foo(y, x, z), xv) ≈
          ForwardDiff.gradient(x -> bar(y, x, z), xv)
    @test ForwardDiff.gradient(x -> foo(y, z, x), xv) ≈
          ForwardDiff.gradient(x -> bar(y, z, x), xv)
    @test ForwardDiff.hessian(x -> foo(x, y, z), xv) ≈
          ForwardDiff.hessian(x -> bar(x, y, z), xv)
    @test ForwardDiff.hessian(x -> foo(y, x, z), xv) ≈
          ForwardDiff.hessian(x -> bar(y, x, z), xv)
    @test ForwardDiff.hessian(x -> foo(y, z, x), xv) ≈
          ForwardDiff.hessian(x -> bar(y, z, x), xv)
  end
end
