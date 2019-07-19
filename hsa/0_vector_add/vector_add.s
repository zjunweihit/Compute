.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 6, "AMD", "AMDGPU"

.text
.p2align 8
.amdgpu_hsa_kernel vector_add

vector_add:

    .amd_kernel_code_t
        enable_sgpr_kernarg_segment_ptr = 1
        is_ptr64 = 1
        compute_pgm_rsrc1_vgprs = 1
        compute_pgm_rsrc1_sgprs = 0
        compute_pgm_rsrc2_user_sgpr = 2
        kernarg_segment_byte_size = 24
        wavefront_sgpr_count = 8
        workitem_vgpr_count = 5
    .end_amd_kernel_code_t

    // initial state:
    //   s[0:1] - kernarg base address
    //   v0 - workitem id

    s_load_dwordx2  s[4:5], s[0:1], 0x10  // load out_ptr into s[4:5] from kernarg
    s_load_dwordx4  s[0:3], s[0:1], 0x00  // load a into s[0:1] and b into s[2:3] from kernarg
    v_lshlrev_b32  v0, 2, v0              // v0 *= 4;
    s_waitcnt     lgkmcnt(0)              // wait for memory reads to finish

    // compute address of corresponding element of out buffer
    // i.e. v[1:2] = &b[workitem_id]
    v_add_co_u32     v1, vcc, s2, v0
    v_mov_b32     v2, s3
    v_addc_co_u32    v2, vcc, v2, 0, vcc

    // compute address of corresponding element of in buffer
    // i.e. v[3:4] = &a[workitem_id]
    v_add_co_u32     v3, vcc, s0, v0
    v_mov_b32     v4, s1
    v_addc_co_u32    v4, vcc, v4, 0, vcc

    flat_load_dword  v1, v[1:2] // load a[workitem_id] into v1
    flat_load_dword  v2, v[3:4] // load b[workitem_id] into v2
    s_waitcnt     vmcnt(0) & lgkmcnt(0) // wait for memory reads to finish

    v_add_f32_e32     v1, v1, v2

    // compute address of corresponding element of out buffer
    // i.e. v[3:4] = &c[workitem_id]
    v_add_co_u32     v3, vcc, s4, v0
    v_mov_b32     v2, s5
    v_addc_co_u32    v4, vcc, v2, 0, vcc

    s_waitcnt     lgkmcnt(0) // wait for permutation to finish

    // store final value in out buffer, i.e. out[workitem_id] = v1
    flat_store_dword  v[3:4], v1

    s_endpgm
