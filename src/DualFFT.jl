__precompile__(true)
module DualFFT

# ---- Import packages ---- #
using AbstractFFTs, ForwardDiff, Base.FFTW

using ForwardDiff: Dual, Partials
using AbstractFFTs: Plan, ScaledPlan

# ---- Imported base functions directly ---- #
import ForwardDiff: value, partials, npartials, valtype, tagtype
import AbstractFFTs: plan_fft, plan_inv, plan_bfft
import Base: A_mul_B!, *
import Base.FFTW: set_timelimit, dims_howmany, unsafe_execute!, cFFTWPlan, r2rFFTWPlan, PlanPtr, FFTWPlan, ScaledPlan, destroy_plan
import Base.DFT: normalization, complexfloat, strides

# ---- Source files ---- #
include("complex_dual.jl")
include("dual_fftw_types.jl")
include("dual_fftw.jl")

# ---- Exported functions ---- #
# export fft

end # module
