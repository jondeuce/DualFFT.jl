# types for fft of complex dual numbers with the FFTW library using the
# AbstractFFTs interface

const DUALFORW = true
const DUALBACK = false
const INPLACE = true

# ---------------------------------------------------------------------------- #
# CplxDualArray: Dummy Array{Complex{Dual}} wrapper for convenience
# ---------------------------------------------------------------------------- #
immutable CplxDualArray{D, Dim} <: DenseArray{Complex{D}, Dim}
    arr
end
CplxDualArray(X::Array{Complex{D}, Dim}) where {D<:Dual, Dim} = CplxDualArray{D, Dim+1}(X)
Base.size(X::CplxDualArray{D,Dim}) where {D<:Dual,Dim} = (numbases(D), size(X.arr)...)
strides(X::CplxDualArray{D,Dim}) where {D<:Dual,Dim} =  NTuple{Dim,Int}([1,2*numbases(D)*cumprod([1,size(X.arr)...])[1:end-1]...])

# ---------------------------------------------------------------------------- #
# DualPlan type (for AbstractFFTs interface)
# ---------------------------------------------------------------------------- #
mutable struct DualPlan{T,forward,inplace} <: Plan{T}
    region
    msize
    pinv::ScaledPlan{T}
    DualPlan{T,forward,inplace}(region,msize) where {T,forward,inplace} = new{T,forward,inplace}(region,msize)
end

# ---------------------------------------------------------------------------- #
# dualFFTWPlan type (for FFTW interface)
# ---------------------------------------------------------------------------- #
mutable struct dualFFTWPlan{T<:fftwReal,K,inplace,Dim} <: FFTWPlan{T,K,inplace}
    plan::PlanPtr
    sz::NTuple{Dim,Int} # size of array on which plan operates (Int tuple)
    osz::NTuple{Dim,Int} # size of output array (Int tuple)
    istride::NTuple{Dim,Int} # strides of input
    ostride::NTuple{Dim,Int} # strides of output
    ialign::Int32 # alignment mod 16 of input
    oalign::Int32 # alignment mod 16 of input
    flags::UInt32 # planner flags
    region::Any # region (iterable) of dims that are transormed
    pinv::ScaledPlan
    function dualFFTWPlan{T,K,inplace,Dim}(plan::PlanPtr,
        flags::Integer, R::Any, x::CplxDualArray{D,Dim},
        y::CplxDualArray{D,Dim}) where {T<:fftwReal,K,inplace,Dim,D<:Dual}

        p = new(plan, size(x), size(y), strides(x), strides(y),
            alignment_of(x.arr), alignment_of(y.arr), flags, R)

        finalizer(p, destroy_plan)
        return p
    end
end

# ---------------------------------------------------------------------------- #
# AbstractFFTs interface functions
# ---------------------------------------------------------------------------- #
AbstractFFTs.plan_fft(  Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALFORW,!INPLACE}(region,size(Z))
AbstractFFTs.plan_fft!( Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALFORW,INPLACE}(region,size(Z))
AbstractFFTs.plan_bfft( Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALBACK,!INPLACE}(region,size(Z))
AbstractFFTs.plan_bfft!(Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALBACK,INPLACE}(region,size(Z))

AbstractFFTs.plan_inv(p::DualPlan{Complex{D},forward,inplace}) where {D<:Dual,forward,inplace} = ScaledPlan(DualPlan{Complex{D},!forward,inplace}(p.region,p.msize),
           normalization(basetype(D), p.msize, p.region))

Base.A_mul_B!(Y::Array{Complex{D}}, p::DualPlan{Complex{D},DUALFORW,inplace}, X::Array{Complex{D}}) where {D<:Dual,inplace} = (Y = copy(X); dualfft!(Y,p.region); return Y)
Base.A_mul_B!(Y::Array{Complex{D}}, p::DualPlan{Complex{D},DUALBACK,inplace}, X::Array{Complex{D}}) where {D<:Dual,inplace} = (Y = copy(X); dualbfft!(Y,p.region); return Y)

Base.:*(p::DualPlan{Complex{D},DUALFORW,!INPLACE}, X::Array{Complex{D}}) where D <: Dual = dualfft(X,p.region)
Base.:*(p::DualPlan{Complex{D},DUALFORW,INPLACE}, X::Array{Complex{D}}) where D <: Dual = (dualfft!(X,p.region); return X)
Base.:*(p::DualPlan{Complex{D},DUALBACK,!INPLACE}, X::Array{Complex{D}}) where D <: Dual = dualbfft(X,p.region)
Base.:*(p::DualPlan{Complex{D},DUALBACK,INPLACE}, X::Array{Complex{D}}) where D <: Dual = (dualbfft!(X,p.region); return X)
