import KernelIntrinsics: _vload_batch, _vstore_batch!, _vload_norebase, _vstore_norebase!, _llvm_barrier

# Metal-compatible barrier (no-op for Metal)
Base.Experimental.@overlay Metal.method_table @inline function _llvm_barrier()
    # No-op for Metal - inline assembly not supported
    nothing
end

# ============================================================================
# Metal GPU Vectorized Memory Access Primitives
# ============================================================================
# These functions provide portable vectorized load/store operations that work
# around Metal shader compiler limitations compared to CUDA/AMDGPU backends.
#
# KEY DESIGN DECISIONS:
#
# 1. SIMD.Vec vs NTuple:
#    Metal's AIR (Apple IR) backend recognizes LLVM vector types <N x T> 
#    (used by SIMD.Vec) to emit vector load/store instructions. However, 
#    Julia's NTuple{N,T} lowers to LLVM array types [N x T], which Metal 
#    scalarizes into N separate scalar loads. This is a fundamental 
#    limitation of Metal's compiler pattern matching versus CUDA's NVPTX 
#    backend, which handles both representations equivalently.
#
# 2. Primitive vs Composite Type Separation:
#    SIMD.Vec{N,T} requires T to be a primitive LLVM scalar type (Integers, 
#    Floats). User-defined structs, even isbits, cannot be used directly 
#    as SIMD.Vec element types. We therefore flatten structs to primitive 
#    chunks (UInt8/UInt16/Float32) for the vector operation, then bitcast 
#    back to the original struct layout. This two-step process is necessary 
#    because Metal lacks CUDA's ability to vectorize arbitrary aggregate types.
#
# 3. @generated Functions and Quote Blocks:
#    Metal's compiler requires all dynamic values (alignment, sizes, types) 
#    to be compile-time constants in the generated LLVM IR. Operations like 
#    `trailing_zeros(N * sizeof(T))` or `Ref(obj)` create runtime calls 
#    (jl_f_trailing_zeros, jl_new_structv, gpu_gc_pool_alloc) that are 
#    unsupported in Metal's device execution environment. By wrapping in 
#    @generated, these computations execute during Julia's compilation phase, 
#    producing literal constants in the quoted code that Metal can inline 
#    directly into shader instructions without runtime overhead.
# ============================================================================


# Determine the optimal primitive chunk type for flattening a struct of size sz.
@generated _chunk_type(sz) = sz == 1 ? :(UInt8) : sz == 2 ? :(UInt16) : :(Float32)

Base.Experimental.@overlay Metal.method_table @inline @generated function _vload_batch(
    A::MtlDeviceArray{T}, idx, ::Val{Nitem}
) where {T, Nitem}
    
    if isprimitivetype(T)
        # For primitive types, we can use SIMD.Vec directly since T is a valid
        # LLVM scalar. This generates a single vector load instruction without
        # intermediate representations.
        sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
        quote
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$Nitem, $T}, 1}, pointer(A))
            return NTuple{$Nitem, $T}(unsafe_load(ptr, idx, Val($sz)))
        end
    else
        # User-defined structs cannot be SIMD.Vec elements. We decompose the
        # struct into primitive chunks matching the total byte size, perform
        # the vector load on those chunks, then reinterpret the loaded bytes
        # as the original struct layout using pointer bitcasting.
        @assert isbitstype(T) "Struct must be isbits for vectorized load"
        struct_sz = sizeof(T)
        ChunkT = _chunk_type(struct_sz)
        n_chunks = div(Nitem * struct_sz, sizeof(ChunkT))
        sz = 0x01 << trailing_zeros(Nitem * struct_sz)
        
        quote
            # Load as vector of primitive chunks (forces vector instruction emission)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$n_chunks, $ChunkT}, 1}, pointer(A))
            raw = unsafe_load(ptr, idx, Val($sz))
            
            # Bitcast: reinterpret loaded chunk vector memory as tuple of structs
            tuptr = Base.unsafe_convert(Ptr{NTuple{$Nitem, $T}}, pointer_from_objref(Ref(raw)))
            unsafe_load(tuptr)
        end
    end
end

Base.Experimental.@overlay Metal.method_table @inline @generated function _vstore_batch!(
    A::MtlDeviceArray{T}, idx, values::NTuple{Nitem, T}
) where {T, Nitem}
    
    if isprimitivetype(T)
        sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
        quote
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$Nitem, $T}, 1}, pointer(A))
            vec = SIMD.Vec{$Nitem, $T}(values)
            unsafe_store!(ptr, vec, idx, Val($sz))
        end
    else
        @assert isbitstype(T)
        struct_sz = sizeof(T)
        ChunkT = _chunk_type(struct_sz)
        n_chunks = div(Nitem * struct_sz, sizeof(ChunkT))
        sz = 0x01 << trailing_zeros(Nitem * struct_sz)
        
        quote
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$n_chunks, $ChunkT}, 1}, pointer(A))
            raw_ptr = Base.unsafe_convert(Ptr{SIMD.Vec{$n_chunks, $ChunkT}}, pointer_from_objref(Ref(values)))
            raw = unsafe_load(raw_ptr)
            unsafe_store!(ptr, raw, idx, Val($sz))
        end
    end
end

Base.Experimental.@overlay Metal.method_table @inline @generated function _vload_norebase(
    A::MtlDeviceArray{T}, idx, ::Val{Nitem}
) where {T, Nitem}
    
    if isprimitivetype(T)
        sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
        quote
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$Nitem, $T}, 1}, offset_ptr)
            return NTuple{$Nitem, $T}(unsafe_load(ptr, 1, Val($sz)))
        end
    else
        @assert isbitstype(T)
        struct_sz = sizeof(T)
        ChunkT = _chunk_type(struct_sz)
        n_chunks = div(Nitem * struct_sz, sizeof(ChunkT))
        sz = 0x01 << trailing_zeros(Nitem * struct_sz)
        
        quote
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$n_chunks, $ChunkT}, 1}, offset_ptr)
            raw = unsafe_load(ptr, 1, Val($sz))
            tuptr = Base.unsafe_convert(Ptr{NTuple{$Nitem, $T}}, pointer_from_objref(Ref(raw)))
            unsafe_load(tuptr)
        end
    end
end

Base.Experimental.@overlay Metal.method_table @inline @generated function _vstore_norebase!(
    A::MtlDeviceArray{T}, idx, values::NTuple{Nitem, T}
) where {T, Nitem}
    
    if isprimitivetype(T)
        sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
        quote
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$Nitem, $T}, 1}, offset_ptr)
            vec = SIMD.Vec{$Nitem, $T}(values)
            unsafe_store!(ptr, vec, 1, Val($sz))
        end
    else
        @assert isbitstype(T)
        struct_sz = sizeof(T)
        ChunkT = _chunk_type(struct_sz)
        n_chunks = div(Nitem * struct_sz, sizeof(ChunkT))
        sz = 0x01 << trailing_zeros(Nitem * struct_sz)
        
        quote
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$n_chunks, $ChunkT}, 1}, offset_ptr)
            raw_ptr = Base.unsafe_convert(Ptr{SIMD.Vec{$n_chunks, $ChunkT}}, pointer_from_objref(Ref(values)))
            raw = unsafe_load(raw_ptr)
            unsafe_store!(ptr, raw, 1, Val($sz))
        end
    end
end
