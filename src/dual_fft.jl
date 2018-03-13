# ----------------------------------------------------- #
# (i)fft for input Z::Vector{Complex{D}}) where D<:Dual #
#     This is faster for vectors despite the extra copy #
#     as all (i)fft's are performed in parallel.        #
# ----------------------------------------------------- #
for (f,d) in ((:fft, :dualfft), (:bfft, :dualbfft))
    f! = Symbol(f,"!")
    d! = Symbol(d,"!")
    @eval begin
        function $d!(Z::Vector{Complex{D}}, region) where D <: Dual
            arr = Array{Complex{valtype(D)},2}(length(Z),npartials(D)+1);
            @inbounds for i = 1:length(Z)
                @inbounds complexdualtorow!(D, arr, Z, i)
            end
            $f!(arr, region)
            @inbounds for i = 1:length(Z)
                @inbounds Z[i] = complexdualfromrow(D, arr, i)
            end
            return nothing
        end
        function $d(Z::Vector{Complex{D}}, region) where D <: Dual
            Y = copy(Z)
            $d!(Y, region)
            return Y
        end
    end
end

end # module
