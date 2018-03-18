# Define the forward, inverse, and backward Fast Fourier Transform on arrays of
# dual numbers from ForwardDiff.Dual using the FFTW library with custom strides

# -------------- #
# FFTW interface #
# -------------- #

@inline floattype(::V) where {V} = error("$V is not an fftwReal")
@inline floattype(::Type{V}) where {V} = error("$V is not an fftwReal")
@inline floattype(::V) where {V<:fftwReal} = V
@inline floattype(::Type{V}) where {V<:fftwReal} = V
@inline floattype(::Dual{T,V,N}) where {T,V,N} = floattype(V)
@inline floattype(::Type{Dual{T,V,N}}) where {T,V,N} = floattype(V)
@inline floattype(::Complex{D}) where {D<:Dual} = floattype(D)
@inline floattype(::Type{Complex{D}}) where {D<:Dual} = floattype(D)

@inline numfloats(::Type{D}) where {D<:Dual} = div(sizeof(D), sizeof(floattype(D)))

# alignment_of(X::Array{Complex{D},Dim}) where {D<:Dual,Dim} = convert(Int32, convert(Int64, pointer(X)) % 16)
alignment_of(X::Array{T,Dim}) where {T<:fftwReal,Dim} = convert(Int32, convert(Int64, pointer(X)) % 16)

for (Tr,Tc,fftw,lib) in ((:Float64,:Complex128,"fftw",FFTW.libfftw),
                         (:Float32,:Complex64,"fftwf",FFTW.libfftwf))
    @eval function Plan_DualFFTW(::Type{D},
        x::Array{$Tr,Dim}, y::Array{$Tr,Dim},
        region, forward, flags::Unsigned, timelimit::Real,
        bitreverse::Bool) where {D<:Dual,$Tr<:fftwReal,Dim}

        set_timelimit($Tr, timelimit)

        # siz = (numfloats(D), size(x)[2:Dim]...)
        siz = size(x)
        dims, howmany = dims_howmany(x, y, [siz...], region)

        PXr = pointer(x)
        PYr = pointer(y)
        PXi = PXr + numfloats(D)*sizeof($Tr)
        PYi = PYr + numfloats(D)*sizeof($Tr)

        # The backward transform is simply the forward transform with real
        # and imaginary parts swapped:
        if forward == FFTW.BACKWARD
            PXr, PXi = PXi, PXr
            PYr, PYi = PYi, PYr
        end

        # generate FFTW plan
        plan = ccall(($(string(fftw,"_plan_guru64_split_dft")),$lib),
                     PlanPtr,
                     (Int32, Ptr{Int}, Int32, Ptr{Int},
                     Ptr{$Tr}, Ptr{$Tr}, Ptr{$Tr}, Ptr{$Tr}, UInt32),
                     size(dims,2), dims, size(howmany,2), howmany,
                     PXr, PXi, PYr, PYi, flags)

        if plan == C_NULL
            error("FFTW could not create plan")
        end

        set_timelimit($Tr, NO_TIMELIMIT)
        p = dualFFTWPlan{$Tr,forward,x===y,Dim}(plan,flags,region,siz,x,y)

        return p
    end
end
function Exec_DualFFTW!(X::Array{Complex{D},Dim}, Y::Array{Complex{D},Dim}, region, forward) where {D<:Dual,Dim}
    Xsiz, Ysiz = size(X), size(Y)
    X = reinterpret(floattype(D), X, (2*numfloats(D), Xsiz...))
    Y = reinterpret(floattype(D), Y, (2*numfloats(D), Ysiz...))

    p = Plan_DualFFTW(D, X, Y, region.+1, forward, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT, false)
    unsafe_execute!(p)

    X = reinterpret(Complex{D}, X, Xsiz)
    Y = reinterpret(Complex{D}, Y, Ysiz)
end

# ------------------------------------------------------------ #
# dual(b)fft for input X::Array{Complex{D},Dim}) where D<:Dual #
# ------------------------------------------------------------ #
for (f,DIRECTION) in ((:dualfft,FFTW.FORWARD), (:dualbfft,FFTW.BACKWARD))
    f! = Symbol(f,"!")
    @eval function $f!(Y::Array{Complex{D},Dim}, X::Array{Complex{D},Dim}, region) where {D<:Dual,Dim}
        Exec_DualFFTW!(X,Y,region,$DIRECTION)
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
