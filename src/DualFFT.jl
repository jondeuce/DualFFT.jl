__precompile__(true)
module DualFFT

# ---- Imported packages ---- #
using AbstractFFTs, ForwardDiff
using ForwardDiff: Dual, Partials
using AbstractFFTs: Plan, ScaledPlan

# ---- Imported base functions ---- #
import ForwardDiff: value, partials, npartials, valtype, tagtype
import AbstractFFTs: plan_fft, plan_inv, plan_bfft
import Base: A_mul_B!, *
import FFTW: set_timelimit, dims_howmany, unsafe_execute!, cFFTWPlan, r2rFFTWPlan, PlanPtr, FFTWPlan, ScaledPlan, destroy_plan
import AbstractFFTs: normalization, complexfloat, strides
import Random
# ---- Source files ---- #
include("complex_dual.jl")
include("abstract_dual_fft.jl")
include("dual_fft.jl")

# ---- Exported functions ---- #
export dual2array, array2dual

end # module
