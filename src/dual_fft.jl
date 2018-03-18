# ------------------------------------------------------------ #
# dual(b)fft for input X::Array{Complex{D},Dim}) where D<:Dual #
# ------------------------------------------------------------ #
for (f,DIRECTION) in ((:fft,FFTW.FORWARD), (:bfft,FFTW.BACKWARD))
    f! = Symbol(f,"!")
    df, df! = Symbol("dual",f), Symbol("dual",f!)
    @eval function $df(X::Array{Complex{D},Dim}, region) where {D<:Dual,Dim}
        y = dual2array(X)
        $f!(y, region.+1)
        Y = array2dual(Complex{D}, y)
        return Y
    end
    @eval function $df!(X::Array{Complex{D},Dim}, region) where {D<:Dual,Dim}
        Y = $df(X, region)
        copy!(X, Y)
        return nothing
    end
end
