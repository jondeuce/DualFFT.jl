# Define the forward, inverse, and backward Fast Fourier Transform on arrays of
# dual numbers from ForwardDiff.Dual using the FFTW library with custom strides

# -------------- #
# FFTW interface #
# -------------- #

# @inline floattype(::Type{D}) where {D<:Dual{T,V,N} where {T,V,N}} = error("floattype(D) is not <: fftwReal ")
# @inline floattype(::Type{D}) where {D<:Dual{T,V,N} where {T,V<:fftwReal,N}} = valtype(D)
# @inline floattype(::Type{D}) where {D<:Dual} = floattype(valtype(D))
# @inline floattype(::Type{Complex{D}}) where {D<:Dual} = floattype(D)

@inline floattype(::V) where {V} = error("$V is not an fftwReal")
@inline floattype(::Type{V}) where {V} = error("$V is not an fftwReal")
@inline floattype(::V) where {V<:fftwReal} = V
@inline floattype(::Type{V}) where {V<:fftwReal} = V
@inline floattype(::Dual{T,V,N}) where {T,V,N} = floattype(V)
@inline floattype(::Type{Dual{T,V,N}}) where {T,V,N} = floattype(V)
@inline floattype(::Complex{D}) where {D<:Dual} = floattype(D)
@inline floattype(::Type{Complex{D}}) where {D<:Dual} = floattype(D)

@inline numfloats(::Type{D}) where {D<:Dual} = div(sizeof(D), sizeof(floattype(D)))

alignment_of(X::Array{Complex{D},Dim}) where {D<:Dual,Dim} = convert(Int32, convert(Int64, pointer(X)) % 16)

strides(X::Array{Complex{D},Dim}) where {D<:Dual,Dim} = NTuple{Dim,Int}(2*numfloats(D)*cumprod([1,size(X)...])[1:end-1])

for (Tr,Tc,fftw,lib) in ((:Float64,:Complex128,"fftw",FFTW.libfftw),
                         (:Float32,:Complex64,"fftwf",FFTW.libfftwf))
    @eval function Plan_DualFFTW(::Type{$Tr},
        X::Array{Complex{D},Dim}, Y::Array{Complex{D},Dim},
        region, forward, flags::Unsigned, timelimit::Real,
        bitreverse::Bool) where {D<:Dual,Dim}

        set_timelimit($Tr, timelimit)
        dims, howmany = dims_howmany(X, Y, [size(X)...], region)
        plantype = dualFFTWPlan{$Tr,forward,X===Y,Dim}
        numplans = numfloats(D)
        plans = Vector{plantype}(numplans)
        for n = 0:numplans-1
            nbytes = sizeof($Tr)
            PXr = pointer(X) + n*nbytes; PXi = PXr + numplans*nbytes
            PYr = pointer(Y) + n*nbytes; PYi = PYr + numplans*nbytes
            # The backward transform is simply the forward transform with real
            # and imaginary parts swapped:
            if forward == FFTW.BACKWARD
                PXr, PXi = PXi, PXr
                PYr, PYi = PYi, PYr
            end
            plan = ccall(($(string(fftw,"_plan_guru64_split_dft")),$lib),
                         PlanPtr,
                         (Int32, Ptr{Int}, Int32, Ptr{Int},
                         Ptr{$Tr}, Ptr{$Tr}, Ptr{$Tr}, Ptr{$Tr}, UInt32),
                         size(dims,2), dims, size(howmany,2), howmany,
                         PXr, PXi, PYr, PYi, flags)
            if plan == C_NULL
                error("FFTW could not create plan") # shouldn't normally happen
            end
            plans[n+1] = plantype(plan,flags,region,X,Y)
        end
        set_timelimit($Tr, NO_TIMELIMIT)
        return plans
    end
end
function Plan_DualFFTW(X::Array{Complex{D},Dim}, Y::Array{Complex{D},Dim},
                       region, forward) where {D<:Dual,Dim}
    return Plan_DualFFTW(floattype(D), X, Y, region, forward, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT, false)
end

# -------------------------------------------------------- #
# dual(b)fft for input X::Array{Complex{D},Dim}) where D<:Dual #
# -------------------------------------------------------- #
for (f,DIRECTION) in ((:dualfft,FFTW.FORWARD), (:dualbfft,FFTW.BACKWARD))
    f! = Symbol(f,"!")
    @eval function $f!(Y::Array{Complex{D},Dim},
                X::Array{Complex{D},Dim}, region) where {D<:Dual,Dim}
        plans = Plan_DualFFTW(X,Y,region,$DIRECTION)
        for p in plans
            unsafe_execute!(p)
        end
        return nothing
    end
    @eval function $f!(X::Array{Complex{D},Dim}, region) where {D<:Dual,Dim}
        $f!(X,X,region)
        return nothing
    end
    @eval function $f(X::Array{Complex{D},Dim}, region) where {D<:Dual,Dim}
        Y = similar(X)
        $f!(Y,X,region)
        return Y
    end
end
