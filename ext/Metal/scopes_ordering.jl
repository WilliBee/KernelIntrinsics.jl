# ext/Metal/scopes_ordering.jl

import KernelIntrinsics: Scope, Ordering
import KernelIntrinsics: Workgroup, Device, System
import KernelIntrinsics: Acquire, Release, AcqRel, SeqCst, Weak, Volatile, Relaxed
import KernelIntrinsics: fence, atomic_load, atomic_store!

import Metal: atomic_store_explicit, atomic_load_explicit

"""
Maps to `air.atomic.fence(i32 flags, i32 order, i32 scope)` (AIR intrinsic).

Memory order, scope, and flags values discovered via:
```
xcrun metal -x metal -E -dM /dev/null | grep METAL_MEMORY
```

```
#define __METAL_MEMORY_FLAGS_DEVICE__ 1
#define __METAL_MEMORY_FLAGS_NONE__ 0
#define __METAL_MEMORY_FLAGS_OBJECT_DATA__ 16
#define __METAL_MEMORY_FLAGS_TEXTURE__ 4
#define __METAL_MEMORY_FLAGS_THREADGROUP_IMAGEBLOCK__ 8
#define __METAL_MEMORY_FLAGS_THREADGROUP__ 2
#define __METAL_MEMORY_ORDER_ACQUIRE__ 2
#define __METAL_MEMORY_ORDER_ACQ_REL__ 4
#define __METAL_MEMORY_ORDER_RELAXED__ 0
#define __METAL_MEMORY_ORDER_RELEASE__ 3
#define __METAL_MEMORY_ORDER_SEQ_CST__ 5
#define __METAL_MEMORY_SCOPE_DEVICE__ 2
#define __METAL_MEMORY_SCOPE_SIMDGROUP__ 4
#define __METAL_MEMORY_SCOPE_THREADGROUP__ 1
#define __METAL_MEMORY_SCOPE_THREAD__ 0
```

**Implementation Details**

Template in metal_atomic.h enforces relaxed-only for atomic load/store:
```cpp
#define METAL_VALID_LOAD_ORDER(O) METAL_ENABLE_IF(O == memory_order_relaxed, ...)
#define METAL_VALID_STORE_ORDER(O) METAL_ENABLE_IF(O == memory_order_relaxed, ...)
```

Also MSL spec 6.15.4.2 states: "For atomic operations other than atomic_thread_fence,
memory_order_relaxed is the only enumeration value."

**Acquire/Release semantics for atomic loads/stores**

We use explicit fences with relaxed atomic operations:
- `Acquire` load: relaxed load → Acquire fence (prevents subsequent ops from moving before load)
- `Release` store: Release fence → relaxed store (prevents previous ops from moving past store)

This approach follows Mojo's implementation of Metal atomics
(https://github.com/modular/modular/blob/main/mojo/stdlib/std/gpu/intrinsics.mojo).

To provide backend-agnostic acquire/release semantics matching CUDA/AMDGPU, 
this is necessary because Metal only supports relaxed atomics for load/store.
"""

# const FENCE_ORDER_TO_METAL = Dict{Type{<:Ordering},UInt32}(
#     Relaxed => UInt32(0),       # __METAL_MEMORY_ORDER_RELAXED__
#     Acquire => UInt32(2),       # __METAL_MEMORY_ORDER_ACQUIRE__
#     Release => UInt32(3),       # __METAL_MEMORY_ORDER_RELEASE__
#     AcqRel  => UInt32(4),       # __METAL_MEMORY_ORDER_ACQ_REL__
#     SeqCst  => UInt32(5),       # __METAL_MEMORY_ORDER_SEQ_CST__
# )

const FENCE_ORDER_TO_METAL = Dict{Type{<:Ordering},UInt32}(
    Relaxed => UInt32(0),       # __METAL_MEMORY_ORDER_RELAXED__
    Acquire => UInt32(2),       # __METAL_MEMORY_ORDER_ACQUIRE__
    Release => UInt32(3),       # __METAL_MEMORY_ORDER_RELEASE__
    AcqRel  => UInt32(4),       # __METAL_MEMORY_ORDER_ACQ_REL__
    SeqCst  => UInt32(5),       # __METAL_MEMORY_ORDER_SEQ_CST__
)

const SCOPE_TO_FLAGS = Dict{Type{<:Scope},UInt32}(
    Workgroup => UInt32(2),     # __METAL_MEMORY_FLAGS_THREADGROUP__
    Device    => UInt32(1),     # __METAL_MEMORY_FLAGS_DEVICE__
    System    => UInt32(1),     # __METAL_MEMORY_FLAGS_DEVICE__ (Metal doesn't have system scope)
)

# Match thread scope to memory scope for proper synchronization
const SCOPE_TO_THREAD_SCOPE = Dict{Type{<:Scope},UInt32}(
    Workgroup => UInt32(1),     # __METAL_MEMORY_SCOPE_THREADGROUP__
    Device    => UInt32(2),     # __METAL_MEMORY_SCOPE_DEVICE__
    System    => UInt32(2),     # __METAL_MEMORY_SCOPE_DEVICE__ (no system scope in Metal)
)

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Relaxed, Acquire, Release, AcqRel, SeqCst]
        flags = SCOPE_TO_FLAGS[ScopeType]
        order = FENCE_ORDER_TO_METAL[OrderType]
        scope = SCOPE_TO_THREAD_SCOPE[ScopeType]

        @eval begin
            """
                fence(::Type{$($ScopeType)}, ::Type{$($OrderType)})

            Metal fence using `air.atomic.fence` intrinsic with correct memory order.

            Uses Metal's memory order values :
            - `Relaxed` (0): No synchronization
            - `Acquire` (2): Acquire fence
            - `Release` (3): Release fence
            - `AcqRel`  (4): Acquire-release fence
            - `SeqCst`  (5): Sequentially consistent fence
            """
            Base.Experimental.@overlay Metal.method_table @inline function fence(
                ::Type{$ScopeType}, ::Type{$OrderType}
            )
                ccall("extern air.atomic.fence", llvmcall, Cvoid,
                      (Cuint, Cuint, Cuint), $(flags), $(order), $(scope))
            end
        end
    end
end

# ============================================================================
# Atomic Load Operations
# ============================================================================
# To achieve acquire semantics matching CUDA/AMDGPU backends, we use a fence
# with a relaxed load.
#
# From MSL spec 6.15.3: "An atomic_thread_fence with memory_order_acquire ordering
# prevents subsequent reads from moving before the load within that scope."
#
# Metal supports Int32, UInt32, and Float32 for atomic operations (MSL spec 2.6).
for ScopeType in [Workgroup, Device, System]
    for T in [Int32, UInt32, Float32]
        # Weak/Relaxed/Volatile: no fence needed
        for OrderType in [Weak, Relaxed, Volatile]
            @eval begin
                Base.Experimental.@overlay Metal.method_table @inline function atomic_load(
                    data::MtlDeviceArray{$T,N,AS},
                    index::Integer,
                    ::Type{$ScopeType},
                    ::Type{$OrderType},
                ) where {N,AS}
                    ptr = pointer(data, index)
                    atomic_load_explicit(ptr)
                end
            end
        end

        @eval begin
            """
                atomic_load(data::MtlDeviceArray{$($T),N,AS}, index::Integer, ::Type{$($ScopeType)}, ::Type{Acquire}) where {N,AS}

            Metal acquire load: relaxed load, Acquire fence.
            Fence after load prevents subsequent operations from moving before the load.
            """
            Base.Experimental.@overlay Metal.method_table @inline function atomic_load(
                data::MtlDeviceArray{$T,N,AS},
                index::Integer,
                ::Type{$ScopeType},
                ::Type{Acquire},
            ) where {N,AS}
                ptr = pointer(data, index)
                val = atomic_load_explicit(ptr)
                fence($ScopeType, Acquire)
                val
            end
        end
    end
end

# ============================================================================
# Atomic Store Operations
# ============================================================================
# To achieve release semantics matching CUDA/AMDGPU backends, we use a fence
# with a relaxed store.
#
# From MSL spec 6.15.3: "An atomic_thread_fence with memory_order_release ordering
# prevents all preceding writes from moving past subsequent stores within
# that scope."
#
# Metal supports Int32, UInt32, and Float32 for atomic operations (MSL spec 2.6).
for ScopeType in [Workgroup, Device, System]
    for T in [Int32, UInt32, Float32]
        # Weak/Relaxed/Volatile: no fence needed
        for OrderType in [Weak, Relaxed, Volatile]
            @eval begin
                Base.Experimental.@overlay Metal.method_table @inline function atomic_store!(
                    data::MtlDeviceArray{$T,N,AS},
                    index::Integer,
                    val,
                    ::Type{$ScopeType},
                    ::Type{$OrderType},
                ) where {N,AS}
                    ptr = pointer(data, index)
                    atomic_store_explicit(ptr, convert($T, val))
                end
            end
        end

        @eval begin
            """
                atomic_store!(data::MtlDeviceArray{$($T),N,AS}, index::Integer, val, ::Type{$($ScopeType)}, ::Type{Release}) where {N,AS}

            Metal release store: Release fence, relaxed store.
            Fence before store prevents previous operations from moving past the store.
            """
            Base.Experimental.@overlay Metal.method_table @inline function atomic_store!(
                data::MtlDeviceArray{$T,N,AS},
                index::Integer,
                val,
                ::Type{$ScopeType},
                ::Type{Release},
            ) where {N,AS}
                fence($ScopeType, Release)
                ptr = pointer(data, index)
                atomic_store_explicit(ptr, convert($T, val))
            end
        end
    end
end
