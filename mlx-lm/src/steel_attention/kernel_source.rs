//! Steel-attention Metal source. Two strings:
//!
//! - [`KERNEL_HEADER`]: file-scope declarations. The first half is the
//!   auto-generated preamble emitted by `make_compiled_preamble.sh`
//!   against `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`,
//!   exposing `BlockLoaderT`, `MMATile`, `BaseMMAFrag`, `tile_matmad`,
//!   `mlx::steel::AttnParams`, and the `MaxOp` / `SumOp` / `ExpSubOp` /
//!   `MulOp` / `DivOp` transforms. The second half is our local
//!   `mlx::steel` namespace alias.
//! - [`KERNEL_SOURCE`]: the kernel body. mlx-rs `metal_kernel` wraps it
//!   in `[[kernel]] void <name>(...)` with each named buffer auto-bound.
//!
//! **A3 scope**: D=128, no causal mask, no explicit mask, no sinks. The
//! body is a port of upstream `attention()` (steel_attention.h:60-476)
//! with `params->*` accesses replaced by the flattened scalar buffers
//! described in [`crate::steel_attention::params::FlatAttnParams`].

pub const KERNEL_HEADER: &str = concat!(
    include_str!(env!("STEEL_ATTENTION_PREAMBLE_PATH")),
    r#"

// --- mlx-lm steel_attention header ----------------------------------
// Steel templates live in `mlx::steel`; alias the namespace so the
// kernel body can use unqualified names.
using namespace mlx::steel;

// QuantBlockLoaderT — mlx-lm extension. Mirrors upstream BlockLoaderT
// (loader.h:142-261) but reads packed `(wq, scales, biases)` triples
// and dequantises into threadgroup `T` memory.
//
// Source layout (matches mlx::quantize affine output):
//  - wq      : uint32_t [B, H, L, BD/PACK]   PACK = 32/BITS
//  - scales  : T         [B, H, L, BD/GROUP]
//  - biases  : T         [B, H, L, BD/GROUP]
// `src_ld` here is the per-token (L axis) stride for the dense BD
// dimension expressed in BD elements — the caller passes `BD` for
// row-major contiguous wq.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short BITS,
    short GROUP,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS  = BCOLS / n_reads,
    short TROWS  = tgp_size / TCOLS>
struct QuantBlockLoaderT {
    STEEL_CONST short n_rows  = (BROWS + TROWS - 1) / TROWS;
    STEEL_CONST short vec_size = n_reads;
    STEEL_CONST short PACK     = 32 / BITS;
    STEEL_CONST uint  MASK     = (1u << BITS) - 1u;

    // Source strides (in elements of BD).
    const int src_ld;       // wq per-token element stride (typically BD).
    const int meta_ld;      // scales/biases per-token element stride (typically BD).
    const int tile_stride;  // bytes per next() step expressed in BD elements.

    // Thread location indices.
    const short thread_idx;
    const short bi;
    const short bj;

    // Threadgroup output + device source bases.
    threadgroup T* dst;
    const device uint32_t* wq;
    const device T* scales;
    const device T* biases;

    METAL_FUNC QuantBlockLoaderT(
        const device uint32_t* wq_,
        const device T* scales_,
        const device T* biases_,
        const int src_ld_,
        const int meta_ld_,
        threadgroup T* dst_,
        ushort simd_group_id [[simdgroup_index_in_threadgroup]],
        ushort simd_lane_id [[thread_index_in_simdgroup]])
        : src_ld(src_ld_),
          meta_ld(meta_ld_),
          tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
          thread_idx(simd_group_id * 32 + simd_lane_id),
          bi(thread_idx / TCOLS),
          bj(vec_size * (thread_idx % TCOLS)),
          dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
          wq(wq_ + bi * (src_ld_ / PACK)),
          scales(scales_ + bi * (meta_ld_ / GROUP)),
          biases(biases_ + bi * (meta_ld_ / GROUP)) {}

    /// Dequantise one (i, j) element at column `col = bj + j` and row
    /// offset `i_row` (in BCOLS elements). Reads wq/scales/biases via
    /// the thread's base pointers offset by `i_row * (src_ld / PACK)`
    /// and `i_row * (meta_ld / GROUP)`.
    METAL_FUNC T dequant_one(short i_row, short j) const {
        short col = bj + j;
        short word_off = col / PACK;
        short slot     = col % PACK;
        short group    = col / GROUP;

        uint w = wq[i_row * (src_ld / PACK) + word_off];
        uint code = (w >> (slot * BITS)) & MASK;
        T sc = scales[i_row * (meta_ld / GROUP) + group];
        T bi_ = biases[i_row * (meta_ld / GROUP) + group];
        return T(code) * sc + bi_;
    }

    METAL_FUNC void load_unsafe() const {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < vec_size; j++) {
                dst[i * kDstStrRow + j * kDstStrCol] = dequant_one(i, j);
            }
        }
    }

    METAL_FUNC void load_safe(short2 src_tile_dim) const {
        src_tile_dim = src_tile_dim - short2(bj, bi);
        if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < BROWS; i += TROWS) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < vec_size; j++) {
                    dst[i * kDstStrRow + j * kDstStrCol] = T(0);
                }
            }
            return;
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
            bool row_ok = i < src_tile_dim.y;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < vec_size; j++) {
                bool ok = row_ok && (j < src_tile_dim.x);
                dst[i * kDstStrRow + j * kDstStrCol] = ok ? dequant_one(i, j) : T(0);
            }
        }
    }

    METAL_FUNC void next() {
        wq     += tile_stride / PACK;
        scales += tile_stride / GROUP;
        biases += tile_stride / GROUP;
    }
};
// --------------------------------------------------------------------
"#,
);

pub const KERNEL_SOURCE: &str = r#"
    // ===== flatten params (mirrors AttnParams field-for-field) =====
    int B_       = int(b_param);    (void)B_;
    int H_       = int(h_param);    (void)H_;
    int D_       = int(d_param);    (void)D_;
    int qL_      = int(q_len_in);
    int kL_      = int(k_len_in);
    int gqa_    = int(gqa_factor);
    float scale_ = float(scale_param);

    int NQ_         = int(nq_param);          (void)NQ_;
    int NK_         = int(nk_param);
    int NQ_aligned_ = int(nq_aligned_param);
    int NK_aligned_ = int(nk_aligned_param);
    int qL_rem_     = int(ql_rem_param);
    int kL_rem_     = int(kl_rem_param);
    int qL_off_     = int(ql_off_param);

    // Strides come in as [int64_t; 3] — Batch, Head, Seq.
    int64_t qs0 = q_strides_in[0], qs1 = q_strides_in[1], qs2 = q_strides_in[2];
    int64_t ks0 = k_strides_in[0], ks1 = k_strides_in[1], ks2 = k_strides_in[2];
    int64_t vs0 = v_strides_in[0], vs1 = v_strides_in[1], vs2 = v_strides_in[2];
    int64_t os0 = o_strides_in[0], os1 = o_strides_in[1], os2 = o_strides_in[2];

    (void)mask[0]; (void)mask_present; // explicit mask path disabled (A3).
    int do_causal_ = int(do_causal_param);

    // ===== compile-time template constants =====
    // mlx-rs injects: T (dtype), BD, BQ, BK, WM, WN.
    using AccumType = float;

    // ===== grid wiring =====
    // grid: (NQ * WM*WN*32, H, B) ; tg: (WM*WN*32, 1, 1).
    // `threadgroup_position_in_grid` is exposed by mlx-rs metal_kernel.
    uint3 tid = threadgroup_position_in_grid;
    uint simd_lane_id  = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    // ===== upstream attention() body — port begins ==================
    // (lines numbered against steel_attention.h v0.31.2 for review.)

    ulong3 tidl{tid.x, tid.y, tid.z};

    const device T* Q_ptr = q
        + tidl.z * qs0
        + tidl.y * qs1
        + tidl.x * uint(BQ) * qs2;

    ulong kv_head_idx = ulong(tid.y) / ulong(gqa_);
    const device T* K_ptr = k + tidl.z * ks0 + kv_head_idx * ks1;
    const device T* V_ptr = v + tidl.z * vs0 + kv_head_idx * vs1;

    device T* O_ptr = o
        + tidl.z * os0
        + tidl.y * os1
        + tidl.x * uint(BQ) * os2;

    // Threadgroup memory layout — Q and KV scratch with 16-byte padding.
    constexpr short padQ = 16 / sizeof(T);
    constexpr short padK = 16 / sizeof(T);
    constexpr short padV = 16 / sizeof(T);

    constexpr short LDQ_tgp = BD + padQ;
    constexpr short LDK_tgp = BK + padK;
    constexpr short LDV_tgp = BD + padV;

    constexpr short tgp_mem_0 = (BK + padK) * (BD);
    constexpr short tgp_mem_1 = BK * (BD + padV);
    constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

    threadgroup T Q_smem[BQ * (BD + padQ)];
    threadgroup T KV_smem[tgp_mem_s];

    threadgroup T* Qs = Q_smem;
    threadgroup T* Ks = KV_smem;
    threadgroup T* Vs = KV_smem;

    using QBlockLoader = BlockLoaderT<
        T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
    using KBlockLoader = BlockLoaderT<
        T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
    using VBlockLoader = BlockLoaderT<
        T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;

    QBlockLoader loader_q(Q_ptr, qs2, Qs, simd_group_id, simd_lane_id);
    KBlockLoader loader_k(K_ptr, ks2, Ks, simd_group_id, simd_lane_id);
    VBlockLoader loader_v(V_ptr, vs2, Vs, simd_group_id, simd_lane_id);

    const AccumType scale_log2e = AccumType(scale_) * M_LOG2E_F;

    constexpr short kFragSize = 8;
    using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
    constexpr int kNWarps = WM * WN;
    static_assert(
        BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
        "BQ must accommodate at least one MMA frag per warp.");
    constexpr int TQ = BQ / (kNWarps * kFragSize);
    constexpr int TK = BK / kFragSize;
    constexpr int TD = BD / kFragSize;
    static_assert(TQ == 1, "A3 assumes TQ == 1");

    MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
    MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
    MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
    MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
    MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
    Otile.clear();

    const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;
    const short tm = kFragSize * TQ * short(simd_group_id);

    const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
    const short Ks_offset = sm * LDK_tgp + sn;
    const short Vs_offset = sm * LDV_tgp + sn;

    constexpr short Qs_tile_stride = kFragSize;
    constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q (one block per TG).
    bool q_tail = (int(tid.x) == NQ_aligned_) && (qL_rem_ > 0);
    if (q_tail) {
        loader_q.load_safe(short2(BD, qL_rem_));
    } else {
        loader_q.load_unsafe();
    }

    constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
    AccumType max_score[kRowsPT];
    AccumType sum_score[kRowsPT] = {0};
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = Limits<AccumType>::finite_min;
    }

    int kb_lim = NK_;
    int kb_min_causal = NK_;
    if (do_causal_) {
        int q_max = (int(tid.x) + 1) * BQ + qL_off_;
        kb_lim = (q_max + BK - 1) / BK;
        kb_lim = min(NK_, kb_lim);
        int q_min = int(tid.x) * BQ + qL_off_;
        q_min = max(0, q_min);
        kb_min_causal = q_min / BK;
    }

    for (int kb = 0; kb < kb_lim; kb++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool k_tail = (kb == NK_aligned_) && (kL_rem_ > 0);
        if (k_tail) {
            loader_k.load_safe(short2(BD, kL_rem_));
        } else {
            loader_k.load_unsafe();
        }

        // S = Q @ K^T
        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
                &Qs[Qs_offset + dd * Qs_tile_stride]);
            Ktile.template load<T, 1, 1, LDK_tgp, 1>(
                &Ks[Ks_offset + dd * Ks_tile_stride]);
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(Stile, Qtile, Ktile, Stile);
        }

        // Apply scale in fp32 (scale_log2e folds the log2(e) constant
        // into the multiplier so the row-reduce can use exp2).
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
            Stile.elems()[ii] *= scale_log2e;
        }

        // Tail k-block: mask out OOB columns with -inf.
        if (k_tail) {
            using stile_t = decltype(Stile);
            using selem_t = typename stile_t::elem_type;
            constexpr auto neg_inf = Limits<selem_t>::finite_min;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kTileRows; i++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < stile_t::kTileCols; j++) {
                    short col_pos = sn + (j * stile_t::kFragCols);
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
                        if ((col_pos + jj) >= kL_rem_) {
                            Stile.frag_at(i, j)[jj] = neg_inf;
                        }
                    }
                }
            }
        }

        // Causal mask: only k-blocks that overlap the Q-block's row
        // range need per-element masking. `kb_min_causal` is the
        // earliest such block; earlier blocks are wholly below the
        // diagonal and left untouched.
        if (do_causal_ && kb >= kb_min_causal) {
            using stile_t = decltype(Stile);
            using selem_t = typename stile_t::elem_type;
            constexpr auto neg_inf = Limits<selem_t>::finite_min;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kTileRows; i++) {
                const int row_pos = int(tid.x) * BQ + qL_off_ + tm + sm
                    + (i * stile_t::kFragRows);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < stile_t::kTileCols; j++) {
                    const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
                        if (row_pos < (col_pos + jj)) {
                            Stile.frag_at(i, j)[jj] = neg_inf;
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_tail) {
            loader_v.load_safe(short2(BD, kL_rem_));
        } else {
            loader_v.load_unsafe();
        }

        // Online softmax: rowmax → expsub → row sum.
        AccumType new_max[kRowsPT];
        AccumType factor[kRowsPT];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            new_max[i] = max_score[i];
        }
        Stile.template row_reduce<MaxOp>(new_max);
        Stile.template row_bin_op<ExpSubOp>(new_max);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            max_score[i] = new_max[i];
        }
        AccumType sum_score_tmp[kRowsPT] = {0};
        Stile.template row_reduce<SumOp>(sum_score_tmp);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
        }
        Otile.template row_bin_op<MulOp>(factor);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    if (BD == 128) {
                        simdgroup_barrier(mem_flags::mem_none);
                    }
                    const short kk = ik * kFragSize;
                    const short dd = id * kFragSize;
                    Vtile.template load<T, 1, 1, LDV_tgp, 1>(
                        &Vs[Vs_offset + kk * LDV_tgp + dd]);
                    if (BD == 128) {
                        simdgroup_barrier(mem_flags::mem_none);
                    }
                    MMAFrag_acc_t::mma(
                        Otile.frag_at(iq, id),
                        Stile.frag_at(iq, ik),
                        Vtile.frag_at(0, 0),
                        Otile.frag_at(iq, id));
                }
            }
        }

        loader_k.next();
        loader_v.next();
    }

    // Normalise + store.
    Otile.template row_bin_op<DivOp>(sum_score);
    threadgroup_barrier(mem_flags::mem_none);

    O_ptr += (tm + sm) * os2 + sn;

    if (q_tail) {
        auto dst_tile_dims = short2(BD - sn, qL_rem_ - (tm + sm));
        if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
            return;
        }
        Otile.template store_safe<T, 1, 1>(O_ptr, os2, dst_tile_dims);
    } else {
        Otile.template store<T, 1, 1>(O_ptr, os2);
    }
"#;

/// Quantised K/V variant of the steel attention kernel body. Identical
/// to [`KERNEL_SOURCE`] modulo the K/V loader type (`QuantBlockLoaderT`
/// instead of `BlockLoaderT`) and the K/V input set (packed `wq` +
/// `scales` + `biases` triples instead of dense tensors). Template
/// constants `BITS` and `GROUP_SIZE` parameterise the dequantisation.
pub const KERNEL_SOURCE_QUANT: &str = r#"
    int B_       = int(b_param);    (void)B_;
    int H_       = int(h_param);    (void)H_;
    int D_       = int(d_param);    (void)D_;
    int qL_      = int(q_len_in);
    int kL_      = int(k_len_in);
    int gqa_    = int(gqa_factor);
    float scale_ = float(scale_param);

    int NQ_         = int(nq_param);          (void)NQ_;
    int NK_         = int(nk_param);
    int NQ_aligned_ = int(nq_aligned_param);
    int NK_aligned_ = int(nk_aligned_param);
    int qL_rem_     = int(ql_rem_param);
    int kL_rem_     = int(kl_rem_param);
    int qL_off_     = int(ql_off_param);

    // Q is dense in this kernel too; only K/V are quantised.
    int64_t qs0 = q_strides_in[0], qs1 = q_strides_in[1], qs2 = q_strides_in[2];
    int64_t os0 = o_strides_in[0], os1 = o_strides_in[1], os2 = o_strides_in[2];

    (void)mask[0]; (void)mask_present;
    int do_causal_ = int(do_causal_param);

    using AccumType = float;

    uint3 tid = threadgroup_position_in_grid;
    uint simd_lane_id  = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    ulong3 tidl{tid.x, tid.y, tid.z};

    const device T* Q_ptr = q
        + tidl.z * qs0
        + tidl.y * qs1
        + tidl.x * uint(BQ) * qs2;

    device T* O_ptr = o
        + tidl.z * os0
        + tidl.y * os1
        + tidl.x * uint(BQ) * os2;

    // Quant K/V base pointers. Layout: wq is [B, H_kv, kL, BD/PACK];
    // scales/biases are [B, H_kv, kL, BD/GROUP]. We derive strides
    // from template constants since the buffers are always contiguous.
    constexpr uint PACK_ = 32u / uint(BITS);
    constexpr uint META_PER_ROW = uint(BD) / uint(GROUP_SIZE);
    constexpr uint WORDS_PER_ROW = uint(BD) / PACK_;

    int H_kv_ = int(H_) / int(gqa_);
    ulong kv_head_idx = ulong(tid.y) / ulong(gqa_);
    ulong kv_batch_base_wq   = tidl.z * ulong(H_kv_) * ulong(kL_) * ulong(WORDS_PER_ROW);
    ulong kv_batch_base_meta = tidl.z * ulong(H_kv_) * ulong(kL_) * ulong(META_PER_ROW);

    const device uint32_t* K_wq_ptr =
        k_wq + kv_batch_base_wq + kv_head_idx * ulong(kL_) * ulong(WORDS_PER_ROW);
    const device T* K_s_ptr =
        k_scales + kv_batch_base_meta + kv_head_idx * ulong(kL_) * ulong(META_PER_ROW);
    const device T* K_b_ptr =
        k_biases + kv_batch_base_meta + kv_head_idx * ulong(kL_) * ulong(META_PER_ROW);

    const device uint32_t* V_wq_ptr =
        v_wq + kv_batch_base_wq + kv_head_idx * ulong(kL_) * ulong(WORDS_PER_ROW);
    const device T* V_s_ptr =
        v_scales + kv_batch_base_meta + kv_head_idx * ulong(kL_) * ulong(META_PER_ROW);
    const device T* V_b_ptr =
        v_biases + kv_batch_base_meta + kv_head_idx * ulong(kL_) * ulong(META_PER_ROW);

    constexpr short padQ = 16 / sizeof(T);
    constexpr short padK = 16 / sizeof(T);
    constexpr short padV = 16 / sizeof(T);

    constexpr short LDQ_tgp = BD + padQ;
    constexpr short LDK_tgp = BK + padK;
    constexpr short LDV_tgp = BD + padV;

    constexpr short tgp_mem_0 = (BK + padK) * (BD);
    constexpr short tgp_mem_1 = BK * (BD + padV);
    constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

    threadgroup T Q_smem[BQ * (BD + padQ)];
    threadgroup T KV_smem[tgp_mem_s];

    threadgroup T* Qs = Q_smem;
    threadgroup T* Ks = KV_smem;
    threadgroup T* Vs = KV_smem;

    using QBlockLoader = BlockLoaderT<
        T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
    using KBlockLoader = QuantBlockLoaderT<
        T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32, BITS, GROUP_SIZE>;
    using VBlockLoader = QuantBlockLoaderT<
        T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32, BITS, GROUP_SIZE>;

    QBlockLoader loader_q(Q_ptr, qs2, Qs, simd_group_id, simd_lane_id);
    KBlockLoader loader_k(
        K_wq_ptr, K_s_ptr, K_b_ptr, BD, BD, Ks, simd_group_id, simd_lane_id);
    VBlockLoader loader_v(
        V_wq_ptr, V_s_ptr, V_b_ptr, BD, BD, Vs, simd_group_id, simd_lane_id);

    const AccumType scale_log2e = AccumType(scale_) * M_LOG2E_F;

    constexpr short kFragSize = 8;
    using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
    constexpr int kNWarps = WM * WN;
    static_assert(
        BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
        "BQ must accommodate at least one MMA frag per warp.");
    constexpr int TQ = BQ / (kNWarps * kFragSize);
    constexpr int TK = BK / kFragSize;
    constexpr int TD = BD / kFragSize;
    static_assert(TQ == 1, "Quant variant assumes TQ == 1");

    MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
    MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
    MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
    MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
    MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
    Otile.clear();

    const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;
    const short tm = kFragSize * TQ * short(simd_group_id);

    const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
    const short Ks_offset = sm * LDK_tgp + sn;
    const short Vs_offset = sm * LDV_tgp + sn;

    constexpr short Qs_tile_stride = kFragSize;
    constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool q_tail = (int(tid.x) == NQ_aligned_) && (qL_rem_ > 0);
    if (q_tail) {
        loader_q.load_safe(short2(BD, qL_rem_));
    } else {
        loader_q.load_unsafe();
    }

    constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
    AccumType max_score[kRowsPT];
    AccumType sum_score[kRowsPT] = {0};
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = Limits<AccumType>::finite_min;
    }

    int kb_lim = NK_;
    int kb_min_causal = NK_;
    if (do_causal_) {
        int q_max = (int(tid.x) + 1) * BQ + qL_off_;
        kb_lim = (q_max + BK - 1) / BK;
        kb_lim = min(NK_, kb_lim);
        int q_min = int(tid.x) * BQ + qL_off_;
        q_min = max(0, q_min);
        kb_min_causal = q_min / BK;
    }

    for (int kb = 0; kb < kb_lim; kb++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool k_tail = (kb == NK_aligned_) && (kL_rem_ > 0);
        if (k_tail) {
            loader_k.load_safe(short2(BD, kL_rem_));
        } else {
            loader_k.load_unsafe();
        }

        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
                &Qs[Qs_offset + dd * Qs_tile_stride]);
            Ktile.template load<T, 1, 1, LDK_tgp, 1>(
                &Ks[Ks_offset + dd * Ks_tile_stride]);
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(Stile, Qtile, Ktile, Stile);
        }

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
            Stile.elems()[ii] *= scale_log2e;
        }

        if (k_tail) {
            using stile_t = decltype(Stile);
            using selem_t = typename stile_t::elem_type;
            constexpr auto neg_inf = Limits<selem_t>::finite_min;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kTileRows; i++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < stile_t::kTileCols; j++) {
                    short col_pos = sn + (j * stile_t::kFragCols);
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
                        if ((col_pos + jj) >= kL_rem_) {
                            Stile.frag_at(i, j)[jj] = neg_inf;
                        }
                    }
                }
            }
        }

        if (do_causal_ && kb >= kb_min_causal) {
            using stile_t = decltype(Stile);
            using selem_t = typename stile_t::elem_type;
            constexpr auto neg_inf = Limits<selem_t>::finite_min;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kTileRows; i++) {
                const int row_pos = int(tid.x) * BQ + qL_off_ + tm + sm
                    + (i * stile_t::kFragRows);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < stile_t::kTileCols; j++) {
                    const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
                        if (row_pos < (col_pos + jj)) {
                            Stile.frag_at(i, j)[jj] = neg_inf;
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_tail) {
            loader_v.load_safe(short2(BD, kL_rem_));
        } else {
            loader_v.load_unsafe();
        }

        AccumType new_max[kRowsPT];
        AccumType factor[kRowsPT];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            new_max[i] = max_score[i];
        }
        Stile.template row_reduce<MaxOp>(new_max);
        Stile.template row_bin_op<ExpSubOp>(new_max);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            max_score[i] = new_max[i];
        }
        AccumType sum_score_tmp[kRowsPT] = {0};
        Stile.template row_reduce<SumOp>(sum_score_tmp);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
        }
        Otile.template row_bin_op<MulOp>(factor);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    if (BD == 128) {
                        simdgroup_barrier(mem_flags::mem_none);
                    }
                    const short kk = ik * kFragSize;
                    const short dd = id * kFragSize;
                    Vtile.template load<T, 1, 1, LDV_tgp, 1>(
                        &Vs[Vs_offset + kk * LDV_tgp + dd]);
                    if (BD == 128) {
                        simdgroup_barrier(mem_flags::mem_none);
                    }
                    MMAFrag_acc_t::mma(
                        Otile.frag_at(iq, id),
                        Stile.frag_at(iq, ik),
                        Vtile.frag_at(0, 0),
                        Otile.frag_at(iq, id));
                }
            }
        }

        loader_k.next();
        loader_v.next();
    }

    Otile.template row_bin_op<DivOp>(sum_score);
    threadgroup_barrier(mem_flags::mem_none);

    O_ptr += (tm + sm) * os2 + sn;

    if (q_tail) {
        auto dst_tile_dims = short2(BD - sn, qL_rem_ - (tm + sm));
        if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
            return;
        }
        Otile.template store_safe<T, 1, 1>(O_ptr, os2, dst_tile_dims);
    } else {
        Otile.template store<T, 1, 1>(O_ptr, os2);
    }
"#;
