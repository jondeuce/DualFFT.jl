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
    X = reinterpret(basetype(D), X, (numbases(D), size(X)...))
    Y = copy(X)
    return Y
end
function dual2array(X::Array{Complex{D},Dim}) where {D<:Dual, Dim}
    Xr = reinterpret(basetype(D), real(X), (numbases(D), size(X)...))
    Xi = reinterpret(basetype(D), imag(X), (numbases(D), size(X)...))
    Z = complex.(Xr, Xi)
    return Z
end

function array2dual(::Type{D}, X::Array{T,Dim}) where {D<:Dual, T, Dim}
    X = reinterpret(D, X, size(X)[2:end])
    Y = copy(X)
    return Y
end
function array2dual(::Type{Complex{D}}, X::Array{Complex{T},Dim}) where {D<:Dual, T, Dim}
    Xr = reinterpret(D, real(X), size(X)[2:end])
    Xi = reinterpret(D, imag(X), size(X)[2:end])
    Z = complex.(Xr, Xi)
    return Z
end

# ----------------------------------------------------- #
# randn(Complex{D}) where partials of D are random, too #
# ----------------------------------------------------- #
@generated function randn_tuple(rng::Random.AbstractRNG, ::Type{NTuple{N,V}}) where {N,V}
    return ForwardDiff.tupexpr(i -> :(randn(rng, V)), N)
end

@generated function randn_tuple(::Type{NTuple{N,V}}) where {N,V}
    return ForwardDiff.tupexpr(i -> :(randn(V)), N)
end

# randn(::Partials)
@inline Base.randn(partials::Partials) = randn(typeof(partials))
@inline Base.randn(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(randn_tuple(NTuple{N,V}))
@inline Base.randn(rng::Random.AbstractRNG, partials::Partials) = randn(rng, typeof(partials))
@inline Base.randn(rng::Random.AbstractRNG, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(randn_tuple(rng, NTuple{N,V}))

# randn(::Dual)
@inline Base.randn(d::Dual) = randn(typeof(d))
@inline Base.randn(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(V), randn(Partials{N,V}))
@inline Base.randn(rng::Random.AbstractRNG, d::Dual) = randn(rng, typeof(d))
@inline Base.randn(rng::Random.AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(rng, V), randn(Partials{N,V}))

# randn(::Complex{Dual})
@inline Base.randn(d::Complex{D}) where D <: Dual = randn(typeof(d))
@inline Base.randn(::Type{Complex{D}}) where D <: Dual = Complex(randn(D), randn(D))
@inline Base.randn(rng::Random.AbstractRNG, d::Complex{D}) where D <: Dual = randn(rng, typeof(d))
@inline Base.randn(rng::Random.AbstractRNG, ::Type{Complex{D}}) where D <: Dual = Complex(randn(rng, D), randn(rng, D))
