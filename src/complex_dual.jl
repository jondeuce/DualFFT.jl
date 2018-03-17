# Define utility for types Complex{D} where D<:Dual

# -------------------- #
# Complex{D} utilities #
# -------------------- #
const RealCplxDual = Union{D,Complex{D}} where D<:Dual
const RealCplxDualOrder2 = Union{D,Complex{D}} where D<:Dual{T,V,N} where {T,V<:Dual,N}

@inline value(z::Complex{D}) where D <: Dual = complex(value(real(z)),value(imag(z)))
@inline partials(z::Complex{D}, i::Int) where D <: Dual = complex(partials(real(z),i),partials(imag(z),i))
@inline partials(z::Complex{D}, i::Int, j::Int) where D <: Dual = complex(partials(real(z),i,j),partials(imag(z),i,j))

@inline npartials(z::Complex{D}) where D <: Dual = npartials(D)
@inline npartials(::Type{Complex{D}}) where D <: Dual = npartials(D)
@inline tagtype(z::Complex{D}) where D <: Dual = tagtype(D)
@inline tagtype(::Type{Complex{D}}) where D <: Dual = tagtype(D)
@inline valtype(z::Complex{D}) where D <: Dual = valtype(D)
@inline valtype(::Type{Complex{D}}) where D <: Dual = valtype(D)

@inline basetype(::V) where {V} = V # fallback
@inline basetype(::Type{V}) where {V} = V # fallback
@inline basetype(::Dual{T,V,N}) where {T,V,N} = V
@inline basetype(::Type{Dual{T,V,N}}) where {T,V,N} = V
@inline basetype(::Dual{T,V,N}) where {T,V<:Dual,N} = basetype(V)
@inline basetype(::Type{Dual{T,V,N}}) where {T,V<:Dual,N} = basetype(V)
@inline basetype(::Complex{D}) where {D<:Dual} = basetype(D)
@inline basetype(::Type{Complex{D}}) where {D<:Dual} = basetype(D)

@inline numbases(::Type{D}) where {D<:Dual} = div(sizeof(D), sizeof(basetype(D)))
@inline numbases(::Type{Complex{D}}) where {D<:Dual} = numbases(D)

# ------------------------- #
# dual2array and array2dual #
# ------------------------- #
function dual2array(X::Array{D,Dim}) where {D<:Dual, Dim}
    Y = reinterpret(basetype(D), X, (numbases(D), size(X)...))
end
function dual2array(X::Array{Complex{D},Dim}) where {D<:Dual, Dim}
    Z = complex.(dual2array(real(X)), dual2array(imag(X)))
end

function array2dual(::Type{D}, X::Array{T,Dim}) where {D<:Dual, T, Dim}
    Y = reinterpret(D, X, size(X)[2:end])
end
function array2dual(::Type{Complex{D}}, X::Array{Complex{T},Dim}) where {D<:Dual, T, Dim}
    Z = complex.(array2dual(D, real(X)), array2dual(D, imag(X)))
end

# ----------------------------------------------------- #
# randn(Complex{D}) where partials of D are random, too #
# ----------------------------------------------------- #
@generated function randn_tuple(rng::AbstractRNG, ::Type{NTuple{N,V}}) where {N,V}
    return ForwardDiff.tupexpr(i -> :(randn(rng, V)), N)
end

@generated function randn_tuple(::Type{NTuple{N,V}}) where {N,V}
    return ForwardDiff.tupexpr(i -> :(randn(V)), N)
end

# randn(::Partials)
@inline Base.randn(partials::Partials) = randn(typeof(partials))
@inline Base.randn(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(randn_tuple(NTuple{N,V}))
@inline Base.randn(rng::AbstractRNG, partials::Partials) = randn(rng, typeof(partials))
@inline Base.randn(rng::AbstractRNG, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(randn_tuple(rng, NTuple{N,V}))

# randn(::Dual)
@inline Base.randn(d::Dual) = randn(typeof(d))
@inline Base.randn(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(V), randn(Partials{N,V}))
@inline Base.randn(rng::AbstractRNG, d::Dual) = randn(rng, typeof(d))
@inline Base.randn(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(rng, V), randn(Partials{N,V}))

# randn(::Complex{Dual})
@inline Base.randn(d::Complex{D}) where D <: Dual = randn(typeof(d))
@inline Base.randn(::Type{Complex{D}}) where D <: Dual = Complex(randn(D), randn(D))
@inline Base.randn(rng::AbstractRNG, d::Complex{D}) where D <: Dual = randn(rng, typeof(d))
@inline Base.randn(rng::AbstractRNG, ::Type{Complex{D}}) where D <: Dual = Complex(randn(rng, D), randn(rng, D))
