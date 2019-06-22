# AbstractFFTs interface

const DUALFORW = true
const DUALBACK = false
const INPLACE = true

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
# AbstractFFTs interface functions
# ---------------------------------------------------------------------------- #
AbstractFFTs.plan_fft(  Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALFORW,!INPLACE}(region,size(Z))
AbstractFFTs.plan_fft!( Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALFORW,INPLACE}(region,size(Z))
AbstractFFTs.plan_bfft( Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALBACK,!INPLACE}(region,size(Z))
AbstractFFTs.plan_bfft!(Z::Array{Complex{D}}, region=1:ndims(Z); kwargs...) where D <: Dual = DualPlan{Complex{D},DUALBACK,INPLACE}(region,size(Z))

AbstractFFTs.plan_inv(p::DualPlan{Complex{D},forward,inplace}) where {D<:Dual,forward,inplace} = ScaledPlan(DualPlan{Complex{D},!forward,inplace}(p.region,p.msize),
           normalization(basetype(D), p.msize, p.region))

LinearAlgebra.mul!(Y::Array{Complex{D}}, p::DualPlan{Complex{D},DUALFORW,inplace}, X::Array{Complex{D}}) where {D<:Dual,inplace} = (copy!(Y,X); dualfft!(Y,p.region); return Y)
LinearAlgebra.mul!(Y::Array{Complex{D}}, p::DualPlan{Complex{D},DUALBACK,inplace}, X::Array{Complex{D}}) where {D<:Dual,inplace} = (copy!(Y,X); dualbfft!(Y,p.region); return Y)

Base.:*(p::DualPlan{Complex{D},DUALFORW,!INPLACE}, X::Array{Complex{D}}) where D <: Dual = dualfft(X,p.region)
Base.:*(p::DualPlan{Complex{D},DUALFORW,INPLACE}, X::Array{Complex{D}}) where D <: Dual = (dualfft!(X,p.region); return X)
Base.:*(p::DualPlan{Complex{D},DUALBACK,!INPLACE}, X::Array{Complex{D}}) where D <: Dual = dualbfft(X,p.region)
Base.:*(p::DualPlan{Complex{D},DUALBACK,INPLACE}, X::Array{Complex{D}}) where D <: Dual = (dualbfft!(X,p.region); return X)
