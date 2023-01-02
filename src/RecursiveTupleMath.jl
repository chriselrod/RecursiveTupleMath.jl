"""
   Implements broadcasting functions that operate elementwise and recursively across `Tuple`s, `NamedTuple`s, `SArray`s, and `ForwardDiff.Dual` numbers.

    The functions are:
    - `badd` (broadcast add)
    - `bsub` (broadcast sub)
    - `bmul` (broadcast mul)
    - `bdiv` (broadcast div)
    - `bmax` (broadcast max)
    - `bmin` (broadcast min)

Note that because it is recursive, `bmul(a, b)` will not necessarilly do the same thing as `map(*, a, b)`. For example, if `a` and `b` are tuples of `SMatrices`, `bmul` will multiply them elementwise, while `map` will multiply them as matrices. This is because `bmul` will call `bmul` on the elements of `a` and `b`, while `map` will call `*` on the elements of `a` and `b`.
"""
module RecursiveTupleMath

using Base: @_inline_meta
export bmax, bmin, badd, bsub, bmul, bdiv

using StaticArrays, ForwardDiff
using ForwardDiff: ≺, @define_binary_dual_op, Dual

@inline lt_fast(a, b) = a < b
@inline lt_fast(a::Float64, b::Float64) = Base.lt_float_fast(a, b)
@inline lt_fast(a::Float32, b::Float32) = Base.lt_float_fast(a, b)
@inline gt_fast(a, b) = lt_fast(b, a)

# @inline le_fast(a, b) = a <= b
# @inline le_fast(a::Float64, b::Float64) = Base.le_float_fast(a, b)
# @inline le_fast(a::Float32, b::Float32) = Base.le_float_fast(a, b)
# @inline ge_fast(a, b) = le_fast(b, a)

@inline badd(a, b) = Base.FastMath.add_fast(a, b)
@inline bsub(a, b) = Base.FastMath.sub_fast(a, b)
@inline bmul(a, b) = Base.FastMath.mul_fast(a, b)
@inline bdiv(a, b) = Base.FastMath.div_fast(a, b)
@inline bmax(a, b) = ifelse(gt_fast(a, b), a, b)
@inline bmin(a, b) = ifelse(lt_fast(a, b), a, b)

for bf ∈ [:bmax, :bmin, :badd, :bsub, :bmul, :bdiv]
  @eval begin
    # fall back to fast
    # terminating case
    @inline $bf(::Number, ::Tuple{}) = ()
    @inline $bf(::Tuple{}, ::Number) = ()

    # broadcast
    @inline $bf(x::Number, y::StaticArray{S}) where {S} = SArray{S}($bf(x, Tuple(y)))
    @inline $bf(y::StaticArray{S}, x::Number) where {S} = SArray{S}($bf(Tuple(y), x))
    @inline $bf(x::Number, y::NamedTuple{S}) where {S} = NamedTuple{S}($bf(x, Tuple(y)))
    @inline $bf(y::NamedTuple{S}, x::Number) where {S} = NamedTuple{S}($bf(Tuple(y), x))

    @inline $bf(a::NamedTuple{S}, b::NamedTuple{S}) where {S} =
      NamedTuple{S}($bf(Tuple(a), Tuple(b)))
    @inline $bf(a::StaticArray{S}, b::StaticArray{S}) where {S} =
      SArray{S}($bf(Tuple(a), Tuple(b)))

    # recurse
    @inline $bf(a::Number, b::Tuple{T,Vararg}) where {T} =
      ($bf(a, first(b)), $bf(a, Base.tail(b))...)
    @inline $bf(b::Tuple{T,Vararg}, a::Number) where {T} =
      ($bf(first(b), a), $bf(Base.tail(b), a)...)
    @inline $bf(a::Tuple{T,Vararg}, b::Tuple{T,Vararg}) where {T} =
      ($bf(first(a), first(b)), $bf(Base.tail(a), Base.tail(b))...)

    @inline $bf(a::Tuple, b::Tuple) = map($bf, a, b)
  end
end
@inline bsub(x::Number) = Base.FastMath.sub_fast(x)
@inline bsub(x::Tuple) = map(bsub, x)
@inline bsub(x::NamedTuple) = map(bsub, x)
@inline bsub(x::StaticArray{S}) where {S} = SArray{S}(map(bsub, Tuple(x)))

@static if VERSION < v"1.7"
  struct Returns{T}
    v::T
  end
  (r::Returns)(_) = r.v
end
@inline function btuple(v, ::Val{D}) where {D}
  ntuple(Returns(v), Val(D))
end

@inline bsub(x::Dual{T}) where {T} = Dual{T}(bsub(x.value), bsub(x.partials.values))

@define_binary_dual_op(
  RecursiveTupleMath.badd,
  begin
    @_inline_meta
    Dual{Txy}(badd(x.value, y.value), badd(x.partials.values, y.partials.values))
  end,
  begin
    @_inline_meta
    Dual{Tx}(badd(x.value, y), x.partials)
  end,
  begin
    @_inline_meta
    Dual{Ty}(badd(x, y.value), y.partials.values)
  end,
)
@define_binary_dual_op(
  RecursiveTupleMath.bsub,
  begin
    @_inline_meta
    Dual{Txy}(bsub(x.value, y.value), bsub(x.partials.values, y.partials.values))
  end,
  begin
    @_inline_meta
    Dual{Tx}(bsub(x.value, y), x.partials)
  end,
  begin
    @_inline_meta
    Dual{Ty}(bsub(x, y.value), bsub(y.partials.values))
  end
)
@define_binary_dual_op(
  RecursiveTupleMath.bmul,
  begin
    @_inline_meta
    Dual{Txy}(
      bmul(x.value, y.value),
      badd(bmul(x.value, y.partials.values), bmul(x.partials.values, y.value)),
    )
  end,
  begin
    @_inline_meta
    Dual{Tx}(bmul(x.value, y), bmul(x.partials.values, y))
  end,
  begin
    @_inline_meta
    Dual{Ty}(bmul(x, y.value), bmul(x, y.partials.values))
  end
)
@define_binary_dual_op(
  RecursiveTupleMath.bdiv,
  begin
    @_inline_meta
    Dual{Txy}(
      bdiv(x.value, y.value),
      bdiv(
        bsub(bmul(x.partials.values, y.value), bmul(x.value, y.partials.values)),
        bmul(y.value, y.value),
      ),
    )
  end,
  begin
    @_inline_meta
    Dual{Tx}(bdiv(x.value, y), bdiv(bmul(x.partials.values, y), bmul(y, y)))
  end,
  begin
    @_inline_meta
    Dual{Ty}(
      bdiv(x, y.value),
      bdiv(bsub(bmul(x, y.partials.values)), bmul(y.value, y.value)),
    )
  end,
)
@define_binary_dual_op(
  RecursiveTupleMath.bmax,
  begin
    @_inline_meta
    cmp = gt_fast(x.value, y.value)
    v = ifelse(cmp, x.value, y.value)
    bcmp = btuple(cmp, Val(length(x.partials)))
    p = map(ifelse, bcmp, x.partials.values, y.partials.values)
    Dual{Txy}(v, p)
  end,
  begin
    @_inline_meta
    cmp = gt_fast(x.value, y)
    v = ifelse(cmp, x.value, y)
    bcmp = btuple(cmp, Val(length(x.partials)))
    bnil = map(zero, x.partials.values)
    p = map(ifelse, bcmp, x.partials.values, bnil)
    Dual{Tx}(v, p)
  end,
  begin
    @_inline_meta
    cmp = gt_fast(x, y.value)
    v = ifelse(cmp, x, y.value)
    bcmp = btuple(cmp, Val(length(y.partials)))
    bnil = map(zero, y.partials.values)
    p = map(ifelse, bcmp, bnil, y.partials.values)
    Dual{Ty}(v, p)
  end,
)
@define_binary_dual_op(
  RecursiveTupleMath.bmin,
  begin
    @_inline_meta
    cmp = lt_fast(x.value, y.value)
    v = ifelse(cmp, x.value, y.value)
    bcmp = btuple(cmp, Val(length(x.partials)))
    p = map(ifelse, bcmp, x.partials.values, y.partials.values)
    Dual{Txy}(v, p)
  end,
  begin
    @_inline_meta
    cmp = lt_fast(x.value, y)
    v = ifelse(cmp, x.value, y)
    bcmp = btuple(cmp, Val(length(x.partials)))
    bnil = map(zero, x.partials.values)
    p = map(ifelse, bcmp, x.partials.values, bnil)
    Dual{Tx}(v, p)
  end,
  begin
    @_inline_meta
    cmp = lt_fast(x, y.value)
    v = ifelse(cmp, x, y.value)
    bcmp = btuple(cmp, Val(length(y.partials)))
    bnil = map(zero, y.partials.values)
    p = map(ifelse, bcmp, bnil, y.partials.values)
    Dual{Ty}(v, p)
  end,
)

end
