using Pkg
Pkg.activate("roc")
using Revise


using KernelAbstractions
using KernelIntrinsics
import KernelIntrinsics as KI
using AMDGPU

using BenchmarkTools

using AMDGPU
# vote_wavefront.jl
using KernelAbstractions
using AMDGPU
using KernelAbstractions
using AMDGPU
import AMDGPU.Device: ballot, activemask

using KernelAbstractions
using AMDGPU
import AMDGPU.Device: ballot


using KernelAbstractions
using AMDGPU
import AMDGPU.Device: ballot_sync, activemask
using KernelAbstractions
using AMDGPU

import AMDGPU.Device: ballot, activelane, wavefrontsize

@kernel function test_vote_all!(out)
    i = @index(Local, Linear)
    out[i] = @vote(All, true)
end

@kernel function test_vote_all!(out)
    i = @index(Local, Linear)
    out[i] = KI._vote(KernelIntrinsics.All, 0xffffffff, true)
end

@kernel function test_vote_all_false!(out)
    i = @index(Local, Linear)
    out[i] = @vote(All, isodd(i))
end

@kernel function test_vote_any!(out)
    i = @index(Local, Linear)
    out[i] = @vote(AnyLane, false)
end

@kernel function test_vote_any_true!(out)
    i = @index(Local, Linear)
    out[i] = @vote(AnyLane, i == 1)
end

backend = ROCBackend()
out = ROCArray{Bool}(undef, 64)

test_vote_all!(backend, 64)(out, ndrange=64)
AMDGPU.synchronize()
@show Array(out)[1]  # expected: true

test_vote_all_false!(backend, 64)(out, ndrange=64)
AMDGPU.synchronize()
@show Array(out)[1]  # expected: false

test_vote_any!(backend, 64)(out, ndrange=64)
AMDGPU.synchronize()
@show Array(out)[1]  # expected: false

test_vote_any_true!(backend, 64)(out, ndrange=64)
AMDGPU.synchronize()
@show Array(out)[1]  # expected: true