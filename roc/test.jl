using Pkg
Pkg.activate("roc")
using Revise


using KernelAbstractions
using KernelIntrinsics
import KernelIntrinsics as KI
using AMDGPU
