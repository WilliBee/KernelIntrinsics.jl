import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot
import KernelIntrinsics: _shfl, _vote

const SHFL_DISPATCH = Dict(
    Up => :simd_shuffle_up,
    Down => :simd_shuffle_down,
    Xor => :simd_shuffle_xor
)

for T in (:Float32, :Int32, :UInt32, :Float16, :Int16, :UInt16)
    for (direction, metal_fname) in SHFL_DISPATCH
        @eval begin
            Base.Experimental.@overlay Metal.method_table @inline _shfl(::Type{$direction}, mask, val::$T, src) =
                $metal_fname(val, src)
        end
    end
end

# =======================================================================================================
# These intrinsics are now exposed as julia function as of https://github.com/JuliaGPU/Metal.jl/pull/744
# Will update when they are part of official release
# =======================================================================================================
simd_shuffle_map = ((Float32, "f32"),
                    (Float16, "f16"),
                    (Int32,   "s.i32"),
                    (UInt32,  "u.i32"),
                    (Int16,   "s.i16"),
                    (UInt16,  "u.i16"),
                    (Int8,    "s.i8"),
                    (UInt8,   "u.i8"))

for (jltype, suffix) in simd_shuffle_map
    @eval begin
        Base.Experimental.@overlay Metal.method_table @inline _shfl(::Type{Idx}, mask, val::$jltype, src) =
            ccall($"extern air.simd_shuffle.$suffix",
                llvmcall, $jltype, ($jltype, Int16), val, src - 0x1)
    end
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{Ballot}, mask, pred)
    # Direct LLVM call to air.simd_ballot
    ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), pred)
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{All}, mask, pred)
    ballot_bits = _vote(Ballot, mask, pred)
    # Direct LLVM call to air.simd_vote_all
    ccall("extern air.simd_vote_all.i64", llvmcall, Bool, (UInt64,), ballot_bits)
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{AnyLane}, mask, pred)
    ballot_bits = _vote(Ballot, mask, pred)
    # Direct LLVM call to air.simd_vote_any
    ccall("extern air.simd_vote_any.i64", llvmcall, Bool, (UInt64,), ballot_bits)
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{Uni}, mask, pred)
    bits = ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), pred)
    active = ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), true)
    # Uniform: all active lanes same value (all true OR all false)
    return (bits == active) || (bits == UInt64(0))
end
