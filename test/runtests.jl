using DualFFT: dual2array, array2dual, valtype, partials, npartials
using ForwardDiff: Dual, Partials, value, partials, order, npartials, valtype
using Base.Test
using BenchmarkTools

# ---------------------------------------------------------------------------- #
# Test (i,b)fft for first and second order duals
# ---------------------------------------------------------------------------- #
D  = Dual{Void, Float64, 2} # First order dual type
D2 = Dual{Void, D, 4} # Second order dual type
D3 = Dual{Void, D2, 3} # Third order dual type
D4 = Dual{Void, D3, 1} # Fourth order dual type

for zdims in ((3,), (5,2), (6,7,5), (7,3,5,8))
zdim = length(zdims)
for zreg in (1, 1:zdim, min(2,zdim):zdim, 1:2:zdim)
for dualtype in (D, D2, D3, D4)
    z = randn(Complex{dualtype}, zdims)
    Z = dual2array(z);
    Zreg = zreg.+1

    @test dual2array(fft(z,zreg))  ≈ fft(Z,Zreg)
    @test dual2array(ifft(z,zreg)) ≈ ifft(Z,Zreg)
    @test dual2array(bfft(z,zreg)) ≈ bfft(Z,Zreg)

    @test (fft!(z,zreg);  fft!(Z,Zreg);  dual2array(z) ≈ Z)
    @test (ifft!(z,zreg); ifft!(Z,Zreg); dual2array(z) ≈ Z)
    @test (bfft!(z,zreg); bfft!(Z,Zreg); dual2array(z) ≈ Z)

    pf = plan_fft(z,zreg); pb = plan_bfft(z,zreg); pi = plan_ifft(z,zreg)
    Pf = plan_fft(Z,Zreg); Pb = plan_bfft(Z,Zreg); Pi = plan_ifft(Z,Zreg)

    @test dual2array(pf*z) ≈ Pf*Z
    @test dual2array(pb*z) ≈ Pb*Z
    @test dual2array(pi*z) ≈ Pi*Z
    @test dual2array(pf\z) ≈ Pf\Z
    @test dual2array(pb\z) ≈ Pb\Z
    @test dual2array(pi\z) ≈ Pi\Z

    @test (w=similar(z); W = similar(Z); A_mul_B!(w,pf,z); A_mul_B!(W,Pf,Z); dual2array(w) ≈ W)
    @test (w=similar(z); W = similar(Z); A_mul_B!(w,pb,z); A_mul_B!(W,Pb,Z); dual2array(w) ≈ W)
    @test (w=similar(z); W = similar(Z); A_mul_B!(w,pi,z); A_mul_B!(W,Pi,Z); dual2array(w) ≈ W)

    pf = plan_fft!(z,zreg); pb = plan_bfft!(z,zreg); pi = plan_ifft!(z,zreg)
    Pf = plan_fft!(Z,Zreg); Pb = plan_bfft!(Z,Zreg); Pi = plan_ifft!(Z,Zreg)

    @test dual2array(pf*z) ≈ Pf*Z
    @test dual2array(pb*z) ≈ Pb*Z
    @test dual2array(pi*z) ≈ Pi*Z
    @test dual2array(pf\z) ≈ Pf\Z
    @test dual2array(pb\z) ≈ Pb\Z
    @test dual2array(pi\z) ≈ Pi\Z
end
end
end

# ---------------------------------------------------------------------------- #
# Timing Tests
# ---------------------------------------------------------------------------- #
# N = 3
# D = Dual{Void, Float64, N}
# # zdims = (19,11)
# # zdims = (1024,1024)
# zdims = (128,128,128)
#
# z = randn(Complex{D}, zdims)
# dims = 1:ndims(z)
# psiz = circshift(1:(ndims(z)+1),1)
# pdims = 2:(ndims(z)+1)
#
# # Time fft for in-place dual fft
# @btime fft!($z) # z acts like (N+1 x zdims) array; fft is along 2:ndims(z)+1
#
# # Time fft of first converting to array (with convert time)
# @btime (Z = dual2array($z); fft!(Z,$dims)) # Z is (zdims x N+1)
# @btime (Z = permutedims(dual2array($z),$psiz); fft!(Z,$pdims)) # Z is (N+1 x zdims)
#
# # Time fft of first converting to array (without convert time)
# Z = dual2array(z); # Z is (zdims x N+1)
# @btime fft!($Z,$dims)
# Z = permutedims(Z,psiz) # Z is (N+1 x zdims)
# @btime fft!($Z,$pdims)

# ---------------------------------------------------------------------------- #
# Test the gradient and hessian of a test function which calls FFTW
# ---------------------------------------------------------------------------- #
# using Calculus
# using ImageFiltering
#
# function f(α)
#     #1D data
#     # N = 2^5
#     N = 32^3
#     z = exp(π/4*im)*linspace(-1.0,1.0,N)
#     β = sin.(cosh.(α))
#     z = imfilter(z, centered(β))
#     z .= cos.(z)
#
#     # #2D data
#     # r = 1:4
#     # N = length(α)
#     # z = [complex(α[mod1(i+j,N)],α[mod1(i*j,N)]) for i in r, j in r]
#
#     # # 3D data
#     # r = 1:32
#     # N = length(α)
#     # z = [complex(α[mod1(i+j,N)],α[mod1(j+k,N)]) for i in r, j in r, k in r]
#
#     # p = plan_fft(z,1:2)
#     # z = p\z; z = p*z;
#     # p = plan_fft!(z,1:2)
#     # p*z; p\z;
#     # p = plan_ifft(z,2:3)
#     # z = p\z; z = p*z;
#     # p = plan_ifft!(z,1:2:3)
#     # p*z; p\z;
#
#     # z = fft(z); z = ifft(z);
#     # fft!(z); ifft!(z);
#
#     # p = plan_bfft(z)
#     # p*z
#
#     @show typeof(z)
#     ifft!(fft(z.*z))
#
#     Σ = √sum(abs2, z)
#     return Σ
# end
# f(α::Number) = f([α])[1]
#
# err(x₁,x₂) = norm(x₁.-x₂) / √maximum( @. max(abs2(x₁), abs2(x₂)) )
#
# function gradients(f,α)
#     ∇₁ = ForwardDiff.gradient(f,α)
#     ∇₂ = Calculus.gradient(f,α)
#     return ∇₁, ∇₂
# end
# gradients(f,α::Number) = gradients(f,[α])
#
# function hessians(f,α)
#     H₁ = ForwardDiff.hessian(f,α)
#     H₂ = Calculus.hessian(f,α)
#     return H₁, H₂
# end
# hessians(f,α::Number) = hessians(f,[α])
#
# function main()
#     for N = 3:5
#         α = randn(N)
#         println("N = $N, ||G1-G2|| = $(err(gradients(f,α)...))")
#         println("N = $N, ||H1-H2|| = $(err(hessians(f,α)...))")
#         # println("f(α):"); @btime f($α)
#         # println("ForwardDiff gradient:"); @btime ForwardDiff.gradient(f,$α)
#         # println("Calculus gradient:"); @btime Calculus.gradient(f,$α)
#     end
# end
# main()
