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

@generated function fill_dual!(A, d::U, i::Int, L::Int) where U <: RealCplxDual
    return quote
        N = $(npartials(U))
        ix = i

        A[ix] = value(d)
        for j in 1:N
            ix += L
            A[ix] = partials(d,j)
        end
    end
end

@generated function fill_dual!(A, d::U, i::Int, L::Int) where U <: RealCplxDualOrder2
    return quote
        M = $(npartials(U))
        N = $(npartials(valtype(U)))
        ix = i

        A[ix] = value(value(d))
        for k in 1:N
            ix += L
            A[ix] = partials(value(d),k)
        end

        for j = 1:M
            ix += L
            A[ix] = value(partials(d,j))
            for k in 1:N
                ix += L
                A[ix] = partials(d,j,k)
            end
        end
    end
end

# @generated function getrealrowpartials(::Type{D}, arr, i) where D <: Dual
#     ex = Expr(:tuple, [:(real(arr[i,$k+1])) for k=1:npartials(D)]...)
#     return :(Partials($ex))
# end
#
# @generated function getimagrowpartials(::Type{D}, arr, i) where D <: Dual
#     ex = Expr(:tuple, [:(imag(arr[i,$k+1])) for k=1:npartials(D)]...)
#     return :(Partials($ex))
# end
#
# @generated function complexdualfromrow(::Type{D}, arr, i) where D <: Dual
#     return :( complex( D(real(arr[i,1]),getrealrowpartials(D,arr,i)), D(imag(arr[i,1]),getimagrowpartials(D,arr,i)) ) )
# end

# ------------------------- #
# dual2array and array2dual #
# ------------------------- #
dual2array_eltype(::Type{Complex{D}}) where D <: Dual = Complex{basetype(D)}
dual2array_eltype(::Type{D}) where D <: Dual = basetype(D)
dual2array_eltype(X::Array{U,Dim}) where {U,Dim} = dual2array_eltype(eltype(X))

function dual2array(X::Array{U,Dim}) where {U<:RealCplxDual, Dim}
    T = dual2array_eltype(X)
    L = length(X)
    N = numbases(U)
    siz = (size(X)..., N)

    a = Array{T}(siz)
    for i in eachindex(X)
        fill_dual!(a, X[i], i, L)
    end

    return a
end

# function array2dual(X::Array{U,Dim}) where {U<:RealCplxDual, Dim}
#
# end

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
