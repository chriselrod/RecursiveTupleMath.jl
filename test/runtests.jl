using Base: Forward
using RecursiveTupleMath, StaticArrays, ForwardDiff
using Test

abc(x::Number) = x
abc(x::SVector{3}) = (a = x[1], b = x[2], c = x[3])

foo0(x, y, z) = sum(min.(z .+ x, (x .- y)) ./ max.(z .- y, (x .* y)))
bar0(x, y, z) = sum(bdiv(bmin(badd(z, x), bsub(x, y)), bmax(bsub(z, y), bmul(x, y))))

foo1(x, y, z) = sum((min.(x, y) ./ max.(x, y)) .* z .- z)
bar1(x, y, z) = sum(bsub(bmul(bdiv(bmin(x, y), bmax(x, y)), z), z))

foo2(x, y, z) = sum((.-(x) ./ y) .+ (z ./ x))
bar2(x, y, z) = sum(badd(bdiv(bsub(x), y), bdiv(z, x)))

struct Cycle{F,Y,Z}
  f::F
  y::Y
  z::Z
  n::Int
end
Cycle(f::F, y::Y, z::Z) where {F,Y,Z} = Cycle(f, y, z, 0)
Cycle(c::Cycle, n::Int) = Cycle(c.f, c.y, c.z, n)
function (f::Cycle)(x)
  y = f.y
  z = f.z
  if f.n == 0
    f.f(x, y, z)
  elseif f.n == 1
    f.f(y, x, z)
  else
    f.f(y, z, x)
  end
end

@testset "RecursiveTupleMath.jl" begin
  for T ∈ (Float64, Float32)
    x = rand(T)
    xv = @SVector rand(T, 3)
    for (f, b) ∈ ((foo0, bar0), (foo1, bar1), (foo2, bar2))
      for y in (rand(T), @SVector(rand(T, 3))), z in (rand(T), @SVector(rand(T, 3)))
        let g = Cycle(f, y, z), h = Cycle(b, y, z), habc = Cycle(b, abc(y), abc(z))
          for n = 1:3
            @test ForwardDiff.derivative(g, x) ≈ ForwardDiff.derivative(h, x)
            @test ForwardDiff.gradient(g, xv) ≈
                  ForwardDiff.gradient(h, xv) ≈
                  ForwardDiff.gradient(habc ∘ abc, xv)
            @test ForwardDiff.hessian(g, xv) ≈
                  ForwardDiff.hessian(h, xv) ≈
                  ForwardDiff.hessian(habc ∘ abc, xv)
            g = Cycle(g, n)
            h = Cycle(h, n)
            habc = Cycle(habc, n)
          end
        end
      end
    end
  end
end
