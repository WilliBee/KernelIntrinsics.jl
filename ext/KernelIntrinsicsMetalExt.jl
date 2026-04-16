module KernelIntrinsicsMetalExt

using Metal
using Metal: LLVMPtr, AS
using LLVM
using LLVM.Interop: @asmcall
using SIMD

import KernelIntrinsics: _warpsize, _laneid
# Import parent module and types


Base.Experimental.@overlay Metal.method_table @inline function _warpsize()
    return Metal.threads_per_simdgroup()
end

Base.Experimental.@overlay Metal.method_table @inline function _laneid()
    return Metal.thread_index_in_simdgroup()
end

include("Metal/device.jl")
include("Metal/scopes_ordering.jl")
include("Metal/shuffle_vote.jl")
include("Metal/vectorization.jl")

end # module KernelIntrinsicsMetalExt
