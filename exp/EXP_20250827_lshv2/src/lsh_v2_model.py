#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSH v2 – Efficient Research Scaffold (rev: reqs-2025-08)
This is a training-capable scaffold with efficient fallbacks. It is not a
production kernel but aims to run realistically and fast enough for ablations.

Key toggles in create_lsh_v2_model(...).
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.utils.checkpoint as ckp

ckp.set_checkpoint_debug_enabled(True)

# ---------- Optional FlashAttention-3 import (safe) ----------
_HAS_FA3 = False
try:
    # flash-attn v3 API is evolving; guard to avoid hard import errors
    from flash_attn.flash_attn_interface import flash_attn_func as _fa3

    _HAS_FA3 = True
except Exception:
    _HAS_FA3 = False


# ---------- Small Helpers ----------
def _maybe_chunked_matmul(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    causal: bool,
    rc_chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Row/Column chunked attention (AttentionRC): OOM-safe fallback.
    q,k,v: [B,H,N,D], mask: [B,1,N,N] or None
    """
    B, H, N, D = q.shape
    out = q.new_zeros(B, H, N, D)
    scale = 1.0 / math.sqrt(D)

    # chunk over rows of Q (and columns of K) to reduce peak mem
    for i0 in range(0, N, rc_chunk_size):
        i1 = min(i0 + rc_chunk_size, N)
        qi = q[:, :, i0:i1]  # [B,H,Ni,D]

        # compute scores in col chunks to bound K*V memory
        acc = None
        for j0 in range(0, N, rc_chunk_size):
            j1 = min(j0 + rc_chunk_size, N)
            kj = k[:, :, j0:j1]  # [B,H,Nj,D]
            vj = v[:, :, j0:j1]  # [B,H,Nj,D]

            scores = torch.einsum("bhid,bhjd->bhij", qi, kj) * scale  # [B,H,Ni,Nj]

            if causal:
                # causal within the full sequence
                # positions i in [i0,i1), j in [j0,j1)
                i_idx = torch.arange(i0, i1, device=q.device).unsqueeze(-1)
                j_idx = torch.arange(j0, j1, device=q.device).unsqueeze(0)
                causal_mask = (j_idx > i_idx).unsqueeze(0).unsqueeze(0)  # [1,1,Ni,Nj]
                scores = scores.masked_fill(causal_mask, float("-inf"))

            if attn_mask is not None:
                scores = scores + attn_mask[:, :, i0:i1, j0:j1]

            probs = F.softmax(scores, dim=-1)  # [B,H,Ni,Nj]
            part = torch.einsum("bhij,bhjd->bhid", probs, vj)  # [B,H,Ni,D]
            acc = part if acc is None else acc + part

        out[:, :, i0:i1] = acc
    return out


def _sdpa(q, k, v, causal: bool, attn_mask: Optional[torch.Tensor] = None):
    # PyTorch SDPA (uses mem-eff kernels where available; CPU-compatible)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal
    )


def _flash3_or_sdpa(q, k, v, causal: bool, attn_mask: Optional[torch.Tensor] = None):
    """
    Try FA3 if available (CUDA); else SDPA. Expects [B,H,N,D].
    """
    if _HAS_FA3 and q.is_cuda and attn_mask is None:
        # flash_attn_func expects [B, N, H, D]
        q_, k_, v_ = [t.transpose(1, 2).contiguous() for t in (q, k, v)]
        out = _fa3(q_, k_, v_, causal=causal)  # [B,N,H,D]
        return out.transpose(1, 2).contiguous()  # [B,H,N,D]
    # SDPA path supports CPU/GPU; attn_mask broadcast ok
    return _sdpa(q, k, v, causal=causal, attn_mask=attn_mask)


# ---------- Global Encoder (Mamba-like or Lightweight Bidir) ----------
class GlobalEncoder(nn.Module):
    def __init__(self, d_model: int, kind: str = "mamba"):
        super().__init__()
        self.kind = kind
        if kind == "mamba":
            self.inp = nn.Linear(d_model, d_model * 2, bias=False)
            self.dw = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            )
            self.gelu = nn.SiLU()
            self.out = nn.Linear(d_model, d_model * 2, bias=False)
        elif kind == "lite_bidir":
            self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.ln = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown global encoder kind {kind}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        if self.kind == "mamba":
            z, gate = self.inp(x).chunk(2, dim=-1)  # [B,N,D],[B,N,D]
            z = self.gelu(self.dw(z.transpose(1, 2)).transpose(1, 2))
            z = z * self.gelu(gate)
            qg, kg = self.out(z).chunk(2, dim=-1)
            return qg, kg
        else:
            y = self.ln(x)
            q, k, v = self.qkv(y).chunk(3, dim=-1)
            # light bidir: mix q/k with a cheap long skip (no attention weights kept)
            qg = self.proj(torch.tanh(q + torch.roll(v, 1, dims=1)))
            kg = self.proj(torch.tanh(k + torch.roll(v, -1, dims=1)))
            return qg, kg


# ---------- Learned Hashing (similarity + orthogonal) ----------
class LearnedHasher(nn.Module):
    """
    Learned hashing that emits two code streams:
      - similarity stream (base + rotations)
      - orthogonal stream (Householder-reflected)

    Bucket-id combination uses XOR-salts in int32 to avoid int64 bottlenecks and
    to remain compatible with AMP / fp8. No gradients are tracked for bucketing.
    """

    def __init__(self, d_model: int, hash_dim: int = 8, n_hashes: int = 4):
        super().__init__()
        self.hash_dim = hash_dim
        self.n_hashes = n_hashes

        self.base = nn.Linear(d_model, hash_dim, bias=False)
        self.rot = nn.Parameter(torch.randn(n_hashes, hash_dim, hash_dim) * 0.05)

        # Householder reflection vector for orthogonal stream
        self.u = nn.Parameter(torch.randn(hash_dim))

        # XOR-salts for bit-combination (int32). One salt per (hash, bit).
        # Fixed once; not trained. Registered as buffer (moved with .to(device))
        salts = torch.randint(
            low=1, high=2**31 - 1, size=(n_hashes, hash_dim), dtype=torch.int32
        )
        self.register_buffer("salts", salts, persistent=False)

    def _householder(self, M: torch.Tensor) -> torch.Tensor:
        # H = I - 2uu^T/||u||^2; reflect to orthogonal subspace
        u = self.u / (self.u.norm() + 1e-6)
        H = torch.eye(M.size(-1), device=M.device, dtype=M.dtype) - 2.0 * torch.outer(
            u, u
        )
        return M @ H

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          sim_codes   : [B, N, n_hashes, hash_dim] (pre-sign activations)
          ortho_codes : [B, N, n_hashes, hash_dim]
        """
        base = self.base(x)  # [B,N,Hd]
        sim, ort = [], []
        for i in range(self.n_hashes):
            r = self.rot[i]
            h = base @ r
            sim.append(h)
            ort.append(self._householder(h))
        return torch.stack(sim, dim=2), torch.stack(ort, dim=2)

    @staticmethod
    def _xor_reduce_int32(x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Custom XOR-reduce along 'dim' in int32 (PyTorch has no .reduce for bitwise_xor).
        x must be int32. Small dims (hash_dim, n_hashes) -> this is fast.
        """
        assert x.dtype == torch.int32
        # Start with first slice along dim
        out = x.select(dim, 0).clone()
        for i in range(1, x.size(dim)):
            out ^= x.select(dim, i)
        return out

    @torch.no_grad()
    def to_bucket_ids(
        self, codes: torch.Tensor, bucket_size: int, seq_len: int
    ) -> torch.Tensor:
        """
        Convert sign(codes) to per-token bucket IDs using XOR-salts (int32).
        Args:
          codes      : [B, N, n_hashes, hash_dim] (float/bf16/fp16)
          bucket_size: int
          seq_len    : int
        Returns:
          bucket_ids : [B, N] (int32 in [0, n_buckets-1])
        """
        # 1) Boolean sign bits in int32
        bits = (codes.detach() >= 0).to(torch.int32)  # [B,N,H,hd]
        B, N, H, Hd = bits.shape
        device = bits.device

        # 2) Broadcast salts: [H,Hd] -> [1,1,H,Hd] and select where bit==1
        salts = self.salts.to(device)  # [H,Hd], int32
        salts_exp = salts.view(1, 1, H, Hd).expand_as(bits)
        masked = torch.where(
            bits.bool(), salts_exp, torch.zeros((), device=device, dtype=torch.int32)
        )  # [B,N,H,Hd] int32

        # 3) XOR-reduce across hash_dim -> per-hash code, then across n_hashes -> final code
        per_hash = self._xor_reduce_int32(masked, dim=3)  # [B,N,H]
        code = self._xor_reduce_int32(per_hash, dim=2)  # [B,N] int32, non-negative

        # 4) Map to bucket range using int32 math (no int64)
        n_buckets = max(1, (seq_len + bucket_size - 1) // bucket_size)  # python int
        bucket_ids = torch.remainder(code, n_buckets).to(torch.int32)  # [B,N] int32

        return bucket_ids


"""Deprecated version (keep for reference)
class LearnedHasher(nn.Module):
    def __init__(self, d_model: int, hash_dim: int = 8, n_hashes: int = 4):
        super().__init__()
        self.hash_dim = hash_dim
        self.n_hashes = n_hashes
        self.base = nn.Linear(d_model, hash_dim, bias=False)
        # rotations per hash to diversify
        self.rot = nn.Parameter(torch.randn(n_hashes, hash_dim, hash_dim) * 0.05)
        # orthogonalization vector for Householder
        self.u = nn.Parameter(torch.randn(hash_dim))

    def _householder(self, M: torch.Tensor) -> torch.Tensor:
        # Householder reflection H = I - 2 u u^T / ||u||^2
        u = self.u / (self.u.norm() + 1e-6)
        H = torch.eye(M.size(-1), device=M.device, dtype=M.dtype) - 2.0 * torch.ger(u, u)
        return M @ H

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        \"""
        Returns:
          sim_codes:  [B,N,n_hashes,hash_dim] (pre-sign activations)
          ortho_codes:[B,N,n_hashes,hash_dim] (orthogonalized)
        \"""
        base = self.base(x)  # [B,N,Hd]
        sim = []
        ort = []
        for i in range(self.n_hashes):
            r = self.rot[i]
            h = base @ r
            sim.append(h)
            ort.append(self._householder(h))
        return torch.stack(sim, dim=2), torch.stack(ort, dim=2)

    @staticmethod
    def to_bucket_ids(codes: torch.Tensor, bucket_size: int, seq_len: int) -> torch.Tensor:
        \"""
        codes: [B,N,n_hashes,hash_dim]; binarize → integer codes → modulo buckets
        \"""
        with torch.no_grad():
            bits = (codes >= 0).to(torch.int32)  # sign hashing
            # pack along last dim
            packed = torch.sum(bits * (2 ** torch.arange(bits.size(-1), device=bits.device)), dim=-1)  # [B,N,n_hashes]
            # reduce n_hashes via xor to a single code
            code = torch.bitwise_xor.reduce(packed, dim=-1)  # [B,N]
            n_buckets = max(1, math.ceil(seq_len / bucket_size))
            bucket_ids = code % n_buckets
        return bucket_ids  # [B,N]
"""


# ---------- Multi-Matrix Factorization + Wavelet Spectral ----------
class SpectralMixer(nn.Module):
    def __init__(self, d_head: int, rank: int = 16, use_wavelet: bool = True):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.Aq = nn.Linear(d_head, rank, bias=False)
        self.Bq = nn.Linear(rank, d_head, bias=False)
        self.Ak = nn.Linear(d_head, rank, bias=False)
        self.Bk = nn.Linear(rank, d_head, bias=False)
        # random spectral gate
        self.rand_gate = nn.Parameter(torch.randn(1, 1, 1, d_head) * 0.01)

    def _haar(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,N,D] -> Haar along N (avg/diff on adjacent pairs)
        B, H, N, D = x.shape
        if N % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), value=0.0)
            N = N + 1
        x = x.view(B, H, N // 2, 2, D)
        avg = x[..., 0, :].add(x[..., 1, :]).mul_(0.5)
        diff = x[..., 0, :].sub(x[..., 1, :]).mul_(0.5)
        return torch.cat([avg, diff], dim=2)  # [B,H,N/2*2,D] ≈ [B,H,N,D]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # low-rank maps
        q = self.Bq(self.Aq(q))
        k = self.Bk(self.Ak(k))
        # optional wavelet on K + random spectral gate
        if self.use_wavelet:
            k = self._haar(k)
        k = k * torch.tanh(1.0 + self.rand_gate)
        return q, k


# ---------- Hierarchical Compression ----------
def compress_seq(x: torch.Tensor, factor: int, mode: str = "avg") -> torch.Tensor:
    # x: [B,H,N,D]; compress along N
    if factor <= 1:
        return x
    B, H, N, D = x.shape
    if mode == "avg":
        # pool in non-overlapping windows
        k = min(factor, N)
        stride = k
        t = x.permute(0, 1, 3, 2).contiguous()  # [B,H,D,N]
        t = F.avg_pool1d(t.view(B * H * D, 1, N), kernel_size=k, stride=stride).view(
            B, H, D, -1
        )
        return t.permute(0, 1, 3, 2).contiguous()
    else:
        # 1D conv as an alternative
        conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=factor,
            stride=factor,
            groups=1,
            bias=False,
        ).to(x.device, x.dtype)
        t = x.permute(0, 1, 3, 2).contiguous().view(B * H, D, N)
        y = conv(t).view(B, H, D, -1).permute(0, 1, 3, 2).contiguous()
        return y


def downsample_ids(bucket_ids: torch.Tensor, target_len: int) -> torch.Tensor:
    # bucket_ids: [B,N]; nearest resize to [B,target_len]
    B, N = bucket_ids.shape
    if target_len == N:
        return bucket_ids
    # nearest "resize" via index mapping
    idx = torch.linspace(0, N - 1, target_len, device=bucket_ids.device)
    idx = idx.round().long()
    return bucket_ids.index_select(1, idx)


# ---------- Long-Term Memory (HashEvict + Compressed) ----------
# --- Replace your LTMemory with this token-capable variant (only the deltas matter) ---
class LTMemory(nn.Module):
    def __init__(self, d_model: int, mem_slots: int = 256, compress: bool = True):
        super().__init__()
        self.mem_slots = mem_slots
        self.compress = compress
        Dc = d_model // 2 if compress else d_model

        self.key_proj = nn.Linear(d_model, Dc, bias=False)
        self.val_proj = nn.Linear(d_model, Dc, bias=False)
        self.to_token = nn.Linear(
            Dc, d_model, bias=False
        )  # <-- projection back to token space

        self.register_buffer("keys", torch.zeros(mem_slots, Dc))
        self.register_buffer("vals", torch.zeros(mem_slots, Dc))
        self.register_buffer("age", torch.zeros(mem_slots))
        self.ptr = 0

    @torch.no_grad()
    def write(self, k_tok: torch.Tensor, v_tok: torch.Tensor):
        if (
            k_tok.size(-1) != self.key_proj.in_features
            or v_tok.size(-1) != self.val_proj.in_features
        ):
            raise RuntimeError(
                f"LTMemory.write expects token-space inputs of dim {self.key_proj.in_features}; "
                f"got {k_tok.size(-1)} and {v_tok.size(-1)}."
            )

        # k_tok, v_tok: [B,N,D] (these are X-space; we'll compress internally)
        k = self.key_proj(k_tok).mean(dim=(0, 1)).detach()
        v = self.val_proj(v_tok).mean(dim=(0, 1)).detach()
        i = self.ptr % self.mem_slots
        self.keys[i], self.vals[i] = k, v
        self.age += 1
        self.age[i] = 0
        self.ptr += 1

    def read_tokens(self, B: int, n_tokens: int, device, dtype) -> torch.Tensor:
        """Return up to n_tokens memory entries as *tokens* to prepend to X."""
        if self.ptr == 0 or n_tokens <= 0:
            return torch.zeros(
                B, 0, self.to_token.out_features, device=device, dtype=dtype
            )
        M = min(self.ptr, self.mem_slots)
        # Simple policy: newest-first (lowest age). You can switch to similarity-based selection.
        top = torch.topk(-self.age[:M], k=min(n_tokens, M)).indices  # new → old
        toks = self.to_token(self.vals.index_select(0, top))  # [m, d_model]
        toks = toks.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
        return toks


"""Deprecated version (keep for reference)
class LTMemory(nn.Module):
    def __init__(self, d_model: int, mem_slots: int = 256, compress: bool = True):
        super().__init__()
        self.mem_slots = mem_slots
        self.compress = compress
        self.key_proj = nn.Linear(d_model, d_model // 2 if compress else d_model, bias=False)
        self.val_proj = nn.Linear(d_model, d_model // 2 if compress else d_model, bias=False)
        self.register_buffer("keys", torch.zeros(mem_slots, self.key_proj.out_features))
        elf.register_buffer("vals", torch.zeros(mem_slots, self.val_proj.out_features))
        self.register_buffer("age",  torch.zeros(mem_slots))
        self.ptr = 0

    @torch.no_grad()
    def write(self, k: torch.Tensor, v: torch.Tensor):
        # k,v: [B,N,Dc] -> write a few mean-pooled slots
        km = k.mean(dim=(0, 1)).detach()   # [Dc]
        vm = v.mean(dim=(0, 1)).detach()   # [Dc]
        self.keys[self.ptr % self.mem_slots] = km
        self.vals[self.ptr % self.mem_slots] = vm
        self.age = self.age + 1
        self.age[self.ptr % self.mem_slots] = 0
        self.ptr += 1

    def read(self, probe_k: torch.Tensor, topk: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        # probe_k: [B,H,N,Dc] -> return memory summary per batch/head as small prefix
        B, H, N, D = probe_k.shape
        if self.ptr == 0:
            # no memory yet
            km = probe_k.new_zeros(B, H, 0, D)
            vm = probe_k.new_zeros(B, H, 0, D)
            return km, vm
        # cosine sim with memory keys
        memK = self.keys[:min(self.ptr, self.mem_slots)]          # [M,D]
        memV = self.vals[:min(self.ptr, self.mem_slots)]          # [M,D]
        # use batch mean key as probe
        probe = probe_k.mean(dim=2)                               # [B,H,D]
        probe = F.normalize(probe, dim=-1)
        memKn = F.normalize(memK, dim=-1)
        sim = torch.einsum("bhd,md->bhm", probe, memKn)           # [B,H,M]
        topv, topi = torch.topk(sim, k=min(topk, sim.size(-1)), dim=-1)
        selK = memK.index_select(0, topi.view(-1)).view(B, H, -1, D)
        selV = memV.index_select(0, topi.view(-1)).view(B, H, -1, D)
        return selK, selV
"""


# ---------- Intra-bucket attention dispatcher ----------
def intra_bucket_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bucket_ids: torch.Tensor,
    causal: bool,
    prefer_flash3: bool,
    rc_chunk_size: int,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    q,k,v: [B,H,N,D]; bucket_ids: [B,N] (same for each head)
    Applies attention independently within each bucket.
    """
    B, H, N, D = q.shape
    out = q.new_empty(B, H, N, D)

    for b in range(B):
        # get segments per bucket (variable sizes)
        ids = bucket_ids[b]  # [N]
        uniq = torch.unique(ids, sorted=True)
        for bu in uniq:
            sel = ids == bu
            if sel.sum() <= 1:
                out[b, :, sel] = v[b, :, sel]
                continue
            qi = q[b : b + 1, :, sel]  # [1,H,Ni,D]
            ki = k[b : b + 1, :, sel]
            vi = v[b : b + 1, :, sel]

            if prefer_flash3:
                oi = _flash3_or_sdpa(qi, ki, vi, causal=causal, attn_mask=None)
            else:
                # Try SDPA first (works CPU/GPU); if mask needed pass it.
                try:
                    oi = _sdpa(qi, ki, vi, causal=causal, attn_mask=None)
                except RuntimeError:
                    oi = _maybe_chunked_matmul(
                        qi,
                        ki,
                        vi,
                        attn_mask=None,
                        causal=causal,
                        rc_chunk_size=rc_chunk_size,
                    )

            out[b, :, sel] = oi[0]
    return out


# ---------- GQA Projections ----------
class GQALatent(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)

    def forward(self, qg: torch.Tensor, kg: torch.Tensor, x: torch.Tensor):
        B, N, D = x.shape
        q = (
            self.q_proj(qg).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        )  # [B,H,N,Dh]
        k = (
            self.k_proj(kg).view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)
        )  # [B,HKV,N,Dh]
        v = self.v_proj(x).view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        return q, k, v


# ---------- LSHv2 Attention Layer (f-part) ----------
class LSHv2AttentionCore(nn.Module):
    """
    LSHv2AttentionCore
    ==================
    A single attention sublayer that:
      1) Prepends *shared register tokens* and (optional) *memory tokens* at the X-stage,
         so the same tokens influence both Q and K.
      2) Builds globally-aware Q/K (Mamba-like or lite-bidir pre-encoder).
      3) Projects Q/K/V with GQA; optionally adds low-rank spectral mixing + Haar wavelet on K.
      4) Hashes queries with two parallel learned streams (similarity vs orthogonality/Householder)
         and forms per-token bucket assignments.
      5) Applies hierarchical per-head sequence compression to the **body only** (prefix stays intact),
         grouping heads by compression factor and computing each group in parallel.
      6) Runs *intra-bucket attention* with an I/O-efficient kernel (FlashAttention-3 if available & CUDA,
         else PyTorch SDPA, else chunked matmul “AttentionRC”), injecting the register tokens into
         every bucket via a boolean `extra_mask` **as K/V only** (no write-back duplication).
      7) Optionally writes a compact summary back to the long-term memory bank.

    Expected external modules (constructed in __init__):
      - GlobalEncoder(d_model, kind): returns globally-aware (qg, kg) from X-stage tokens.
      - GQALatent(d_model, n_heads, n_kv_heads): returns (q, k, v) as [B,H,N,Dh] with GQA replication.
      - LearnedHasher(d_model, hash_dim=8, n_hashes=4): forward(x)->(sim_codes, ortho_codes);
        to_bucket_ids(codes, bucket_size, seq_len)->[B,N] bucket assignments (prefer int32).
      - SpectralMixer(d_head, rank, use_wavelet): (q,k)->(q',k') low-rank & wavelet-enhanced features.
      - LTMemory(d_model, mem_slots, compress):
          read_tokens(B, n, device, dtype)->[B,n,D]; write(k_tok, v_tok).

    Notes:
      * “Prefix” = registers + memory tokens. Prefix is never compressed; it is present in every head.
      * “Body”   = original sequence tokens. Body may be compressed per-head.
      * Registers (first R tokens) are injected into every bucket **as K/V only** via `extra_mask`
        (Q does not include registers; avoids write-back contention).
      * Memory tokens are prepended to X and hashed like normal tokens.
      * Causal masking is applied inside each bucket if `causal=True`.
      * Per-head compression factors are repeated round-robin from [1, s, s^2] (s=compression_base).
      * Bucket IDs are computed on the augmented sequence (prefix + body), then downsampled for compressed views.
      * Memory write uses mean-pooled token space over the augmented sequence.
      * IDs/bucket bookkeeping uses int32 to be friendly with AMP/fp8 training.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        bucket_size: int,
        global_kind: str = "mamba",
        prefer_flash3: bool = True,
        rc_chunk_size: int = 2048,
        compression_base: int = 4,
        use_spectral: bool = True,
        use_memory: bool = True,
        n_register: int = 4,
        n_mem_tokens: int = 4,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.bucket_size = int(bucket_size)
        self.prefer_flash3 = prefer_flash3
        self.rc_chunk_size = int(rc_chunk_size)
        self.use_spectral = use_spectral
        self.n_register = int(n_register)
        self.n_mem_tokens = int(n_mem_tokens)

        # --- External subsystems (must be provided by the surrounding codebase) ---
        self.global_enc = GlobalEncoder(
            d_model, kind=global_kind
        )  # -> qg, kg  [B,N',D]
        self.normQ = RMSNorm(d_model, eps=1e-6)  # RMSNorm for qg
        self.normK = RMSNorm(d_model, eps=1e-6)  # RMSNorm for kg
        self.gqa = GQALatent(d_model, n_heads, n_kv_heads)  # -> q,k,v  [B,H,N',Dh]
        self.hasher = LearnedHasher(
            d_model, hash_dim=8, n_hashes=4
        )  # learned 2-stream hashing
        self.spectral = SpectralMixer(
            d_head=d_model // n_heads, rank=16, use_wavelet=use_spectral
        )

        # Memory (optional)
        self.mem = (
            LTMemory(d_model, mem_slots=256, compress=True) if use_memory else None
        )

        # --- Shared register tokens (affect Q & K because they are added at X-stage) ---
        self.register_tokens = nn.Parameter(
            torch.randn(self.n_register, d_model) * 0.02
        )
        self.reg_ln = nn.LayerNorm(d_model)

        # Hierarchical per-head compression pattern
        pattern = [1, max(1, compression_base), max(1, compression_base**2)]
        self.comp_factors = [pattern[h % len(pattern)] for h in range(n_heads)]

        # Output projection + post-norm
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

        # Try to cache FlashAttention-3 handle if available (CUDA-only)
        self._fa3 = None
        if self.prefer_flash3:
            try:
                from flash_attn.flash_attn_interface import flash_attn_func as _fa3

                self._fa3 = _fa3
            except Exception:
                self._fa3 = None  # Fallback will be SDPA/RC

    # -------------------------------------------------------------------------
    # public forward
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        x : [B, N, D]
        return: [B, N, D]
        """
        B, N, D = x.shape
        device, dtype = x.device, x.dtype

        # (A) Build X-stage prefix = [register_tokens, memory_tokens], then X_aug = [prefix, X]
        reg = (
            self.reg_ln(self.register_tokens).unsqueeze(0).expand(B, -1, -1)
        )  # [B, R, D]
        mem_toks = (
            self.mem.read_tokens(B, self.n_mem_tokens, device, dtype)  # [B, M, D]
            if (self.mem is not None and self.n_mem_tokens > 0)
            else x.new_zeros(B, 0, D)
        )
        x_aug = torch.cat([reg, mem_toks, x], dim=1)  # [B, N', D]
        prefix_len = x_aug.size(1) - N
        R = self.n_register

        # (B) Global encoder (e.g., Mamba-like) -> qg, kg
        qg, kg = self.global_enc(x_aug)  # [B, N', D]
        qg = self.normQ(qg)
        kg = self.normK(kg)

        # (C) GQA projections
        q, k, v = self.gqa(qg, kg, x_aug)  # [B, H, N', Dh]

        # (D) Optional spectral mixing
        if self.use_spectral:
            q, k = self.spectral(q, k)

        # (E) Learned hashing streams on qg (augmented)
        sim_codes, ortho_codes = self.hasher(qg)  # streamed hash reps
        bid_sim = self.hasher.to_bucket_ids(sim_codes, self.bucket_size, q.size(2)).to(
            torch.int32
        )  # [B, N']
        bid_ort = self.hasher.to_bucket_ids(
            ortho_codes, self.bucket_size, q.size(2)
        ).to(torch.int32)  # [B, N']

        # (F) Build static K/V register mask to inject registers into every bucket
        #     Note: we will include registers **only** as K/V; Q stays bucket-local.
        extra_kv_mask_full = None
        if R > 0:
            extra_kv_mask_full = torch.zeros(q.size(2), dtype=torch.bool, device=device)
            extra_kv_mask_full[:R] = True

        # (G) Group heads by compression factor, compute each group in parallel
        uniq_factors = sorted(set(self.comp_factors))
        out_heads = q.new_zeros(q.shape)  # [B, H, N', Dh]

        for f in uniq_factors:
            heads = [h for h, ff in enumerate(self.comp_factors) if ff == f]
            if not heads:
                continue

            # Slice to group tensors: [B, Hg, N', Dh]
            qg_group = q[:, heads]
            kg_group = k[:, heads]
            vg_group = v[:, heads]

            # Body-only compression; prefix [0:prefix_len) unchanged
            qgc = self._compress_body_group(qg_group, f, prefix_len)  # [B,Hg,Nc,Dh]
            kgc = self._compress_body_group(kg_group, f, prefix_len)
            vgc = self._compress_body_group(vg_group, f, prefix_len)

            # Align lengths (defensive)
            Nc = min(qgc.size(2), kgc.size(2), vgc.size(2))
            if qgc.size(2) != Nc:
                qgc = qgc[:, :, :Nc]
            if kgc.size(2) != Nc:
                kgc = kgc[:, :, :Nc]
            if vgc.size(2) != Nc:
                vgc = vgc[:, :, :Nc]

            # Downsample bucket ids for the compressed view (body only)
            bids_sim_c = self._downsample_ids_body(
                bid_sim, Nc, prefix_len, factor=f
            )  # [B, Nc] int32
            bids_ort_c = self._downsample_ids_body(
                bid_ort, Nc, prefix_len, factor=f
            )  # [B, Nc] int32

            # Build K/V register mask aligned to Nc (only the first R positions are registers)
            extra_mask_c = None
            if extra_kv_mask_full is not None:
                extra_mask_c = torch.zeros(Nc, dtype=torch.bool, device=device)
                # registers are always within prefix and prefix is uncompressed
                extra_mask_c[: min(R, Nc)] = True

            # Split subgroups by parity: even→similarity, odd→orthogonal
            heads_even = [h for h in heads if (h % 2 == 0)]
            heads_odd = [h for h in heads if (h % 2 == 1)]

            # EVEN heads (similarity stream)
            if heads_even:
                idx = torch.tensor([heads.index(h) for h in heads_even], device=device)
                out_even = self._intra_bucket_attn(
                    qgc[:, idx],
                    kgc[:, idx],
                    vgc[:, idx],
                    bucket_ids=bids_sim_c,
                    causal=causal,
                    extra_mask=extra_mask_c,
                )
                out_even = self._maybe_upsample_back(
                    out_even, N_target=q.size(2), prefix_len=prefix_len
                )
                out_heads[:, heads_even] = out_even

            # ODD heads (orthogonal stream)
            if heads_odd:
                idx = torch.tensor([heads.index(h) for h in heads_odd], device=device)
                out_odd = self._intra_bucket_attn(
                    qgc[:, idx],
                    kgc[:, idx],
                    vgc[:, idx],
                    bucket_ids=bids_ort_c,
                    causal=causal,
                    extra_mask=extra_mask_c,
                )
                out_odd = self._maybe_upsample_back(
                    out_odd, N_target=q.size(2), prefix_len=prefix_len
                )
                out_heads[:, heads_odd] = out_odd

        # (H) Merge heads, project, post-norm
        y = out_heads.transpose(1, 2).contiguous().view(B, q.size(2), self.d_model)
        y = self.out_proj(y)
        y = self.ln(y)

        # (I) Memory write (detach to keep it lightweight unless you need grads)
        # ─ skipped while `self.training` is True so that
        #     gradient-checkpoint re-computations see identical module state.
        #     (The pointer advance inside `LTMemory.write` is a side effect that
        #      otherwise changes `prefix_len`, shifting every downstream shape
        #      by ±1 and triggering the metadata-mismatch you observed.)
        if self.mem is not None:
            with torch.no_grad():
                self.mem.write(qg, x)  # [B,N',D]

        # (J) Return only the original body (drop prefix)
        return y[:, -N:, :]

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    def _compress_body_group(
        self, x: torch.Tensor, factor: int, prefix_len: int
    ) -> torch.Tensor:
        """
        Left-causal body compression by non-overlapping windows of length `factor`,
        with the last (partial) window padded by repeating the last real token.
        x : [B, H, N, Dh]  where N = prefix_len + N_body
        returns [B, H, prefix_len + ceil(N_body/factor), Dh]
        """
        if factor <= 1:
            return x

        B, H, N, Dh = x.shape
        if prefix_len >= N:
            return x  # no body

        pre = x[:, :, :prefix_len]  # [B,H,prefix_len,Dh]
        body = x[:, :, prefix_len:]  # [B,H,N_body,Dh]
        N_body = body.size(2)

        Nc_body = (N_body + factor - 1) // factor
        pad = Nc_body * factor - N_body
        if pad > 0:
            last = body[:, :, -1:, :]  # [B,H,1,Dh]
            body = torch.cat([body, last.expand(B, H, pad, Dh)], dim=2)

        body = body.view(B, H, Nc_body, factor, Dh).mean(dim=3)  # [B,H,Nc_body,Dh]
        return torch.cat([pre, body], dim=2)

    def _downsample_ids_body(
        self, ids: torch.Tensor, target_len: int, prefix_len: int, factor: Optional[int]
    ) -> torch.Tensor:
        """
        Downsample bucket ids to match compressed length. Keeps prefix untouched.
        ids        : [B, N']  (int32/int64 ok; returns int32)
        target_len : desired total length after compression (prefix + Nc_body)
        prefix_len : int
        factor     : compression factor for body; if >1, take left window starts.
        """
        B, Np = ids.shape
        if target_len == Np:
            return ids.to(torch.int32)

        pre = (
            ids[:, :prefix_len]
            if prefix_len > 0
            else ids.new_zeros(B, 0, dtype=ids.dtype)
        )
        body = ids[:, prefix_len:]  # [B, N_body]
        N_body = body.size(1)
        target_body = max(0, target_len - prefix_len)

        if target_body == N_body:
            return ids.to(torch.int32)
        if target_body == 0:
            return pre.to(torch.int32)

        if factor is not None and factor > 1:
            starts = (
                torch.arange(target_body, device=ids.device, dtype=torch.int32) * factor
            )
            starts = torch.clamp(starts, max=N_body - 1)
            body_ds = body.index_select(1, starts.to(torch.long)).to(torch.int32)
            return torch.cat([pre.to(torch.int32), body_ds], dim=1)

        # Fallback nearest for safety (rare)
        body_f = body.to(torch.float32).unsqueeze(1)  # [B,1,N_body]
        body_ds = F.interpolate(
            body_f, size=target_body, mode="nearest"
        )  # [B,1,target_body]
        body_i32 = body_ds.squeeze(1).to(torch.int32)
        return torch.cat([pre.to(torch.int32), body_i32], dim=1)

    def _maybe_upsample_back(
        self, x_c: torch.Tensor, N_target: int, prefix_len: int
    ) -> torch.Tensor:
        """
        Causal, left-anchored inverse of _compress_body_group.
        x_c: [B, H, Nc, Dh]  where Nc = prefix_len + Nc_body
        returns: [B, H, N_target, Dh]  with prefix preserved verbatim.
        """
        B, H, Nc, Dh = x_c.shape
        if Nc == N_target:
            return x_c

        if Nc <= prefix_len:
            pad_len = N_target - Nc
            pad = x_c.new_zeros(B, H, pad_len, Dh)
            return torch.cat([x_c, pad], dim=2)

        pre = x_c[:, :, :prefix_len]  # [B,H,prefix_len,Dh]
        body_c = x_c[:, :, prefix_len:]  # [B,H,Nc_body,Dh]
        Nc_body = body_c.size(2)
        Nt_body = N_target - prefix_len

        t = torch.arange(Nt_body, device=x_c.device, dtype=torch.int32)  # [Nt_body]
        idx = torch.div(t * Nc_body, Nt_body, rounding_mode="floor")  # [Nt_body]
        idx = idx.clamp_(0, Nc_body - 1).to(torch.long)

        body_up = body_c.index_select(2, idx)  # [B,H,Nt_body,Dh]
        return torch.cat([pre, body_up], dim=2)

    # ---- kernels -------------------------------------------------------------
    def _attention_kernel(
        self,
        q: torch.Tensor,  # [B,H,L,D]
        k: torch.Tensor,  # [B,H,S,D]
        v: torch.Tensor,  # [B,H,S,D]
        causal: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Choose attention kernel: FlashAttention-3 (CUDA only) → SDPA → chunked RC.
        Accepts different query length L and key/value length S.
        Returns [B,H,L,D].
        """
        # FlashAttention-3 path (CUDA only; no external mask)
        if (self._fa3 is not None) and q.is_cuda and (attn_mask is None):
            # flash-attn expects [B,L,H,D] and [B,S,H,D]
            q_ = q.transpose(1, 2).contiguous()  # [B,L,H,D]
            k_ = k.transpose(1, 2).contiguous()  # [B,S,H,D]
            v_ = v.transpose(1, 2).contiguous()  # [B,S,H,D]
            out = self._fa3(q_, k_, v_, causal=causal)  # [B,L,H,D]
            return out.transpose(1, 2).contiguous()  # [B,H,L,D]

        # SDPA (CPU/GPU)
        try:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal
            )
        except RuntimeError:
            # Chunked fallback
            return self._attention_rc(q, k, v, attn_mask, causal, self.rc_chunk_size)

    def _attention_rc(
        self,
        q: torch.Tensor,  # [B,H,L,D]
        k: torch.Tensor,  # [B,H,S,D]
        v: torch.Tensor,  # [B,H,S,D]
        attn_mask: Optional[torch.Tensor],
        causal: bool,
        chunk: int,
    ) -> torch.Tensor:
        """
        OOM-safe row/column chunked attention. Supports L != S.
        """
        B, H, L, D = q.shape
        S = k.size(2)
        out = q.new_zeros(B, H, L, D)
        scale = 1.0 / math.sqrt(D)

        for i0 in range(0, L, chunk):
            i1 = min(i0 + chunk, L)
            qi = q[:, :, i0:i1]  # [B,H,Li,D]
            acc = None
            for j0 in range(0, S, chunk):
                j1 = min(j0 + chunk, S)
                kj = k[:, :, j0:j1]  # [B,H,Sj,D]
                vj = v[:, :, j0:j1]  # [B,H,Sj,D]
                scores = torch.einsum("bhid,bhjd->bhij", qi, kj) * scale  # [B,H,Li,Sj]

                if causal:
                    # mask future: positions j beyond i in original time index space
                    # We conservatively apply when Li and Sj correspond to same scale; this is fine for buckets.
                    i_idx = torch.arange(i0, i1, device=q.device).unsqueeze(-1)
                    j_idx = torch.arange(j0, j1, device=q.device).unsqueeze(0)
                    c_mask = (j_idx > i_idx).unsqueeze(0).unsqueeze(0)  # [1,1,Li,Sj]
                    scores = scores.masked_fill(c_mask, float("-inf"))

                if attn_mask is not None:
                    scores = scores + attn_mask[:, :, i0:i1, j0:j1]

                probs = F.softmax(scores, dim=-1)
                part = torch.einsum("bhij,bhjd->bhid", probs, vj)  # [B,H,Li,D]
                acc = part if acc is None else acc + part

            out[:, :, i0:i1] = acc
        return out

    # ---- bucketed attention --------------------------------------------------
    def _intra_bucket_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bucket_ids: torch.Tensor,
        causal: bool,
        extra_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fixed-shape, padded per-bucket attention to be checkpoint-safe.

        q,k,v       : [B, H, N, Dh]
        bucket_ids  : [B, N] (int32) – ids produced on the **compressed** timeline already
        extra_mask  : [N] bool – marks the first R register positions (prefix) to inject into every bucket
        returns     : [B, H, N, Dh]
        """
        B, H, N, Dh = q.shape
        device = q.device
        out = q.new_zeros(B, H, N, Dh)

        # --- Fixed number of buckets (shape-invariant across replays) ---
        # We don't trust ids.max() across replays; use seq_len + bucket_size to upper-bound.
        # The "+ 2" leaves space for sparse end buckets and keeps loop trip-count constant.
        num_buckets = (max(N, 1) + self.bucket_size - 1) // self.bucket_size + 2

        # --- Fixed segment width per bucket: Kmax = bucket_size + (#registers included) ---
        R = int(self.n_register) if (extra_mask is not None) else 0
        Kmax = min(N, self.bucket_size + R)  # cap at N

        # Precompute per-position absolute indices for causal mask construction
        abs_idx = torch.arange(N, device=device, dtype=torch.int32)  # [N]

        # Expand a shared "register mask" (prefix injection) aligned to current N
        # If extra_mask length != N (e.g., after compression), nearest expand already happened upstream.
        inject_regs = extra_mask
        if inject_regs is not None and inject_regs.numel() != N:
            em = inject_regs.to(torch.float32).view(1, 1, -1)
            em = F.interpolate(em, size=N, mode="nearest")
            inject_regs = em.view(N).to(torch.bool)

        # Constant-time loop count ensures checkpoint reproducibility
        for b in range(B):
            ids_b = bucket_ids[b]  # [N], int
            # Per-bucket selection (padded to Kmax)
            for bu in range(num_buckets):
                # Boolean mask for this bucket
                sel = ids_b == bu
                # Inject shared register tokens into every bucket if requested
                if inject_regs is not None:
                    sel = sel | inject_regs

                # If bucket is empty even after injection, create a dummy index {0} and mask it as invalid
                if not torch.any(sel):
                    # skip compute but keep shapes by writing zeros
                    continue

                # Absolute positions selected for this bucket
                idx = torch.where(sel)[0]  # [K_real]
                K_real = int(idx.numel())
                # Pad/truncate to Kmax for shape stability
                if K_real >= Kmax:
                    idx_pad = idx[:Kmax]
                    valid_len = Kmax
                else:
                    pad_len = Kmax - K_real
                    pad_val = idx[-1]  # repeat last valid position
                    pad = pad_val.repeat(pad_len)
                    idx_pad = torch.cat([idx, pad], dim=0)
                    valid_len = K_real

                # Gather q/k/v for these positions -> [B=1, H, Kmax, Dh]
                qb = q[b : b + 1, :, idx_pad]  # [1,H,Kmax,Dh]
                kb = k[b : b + 1, :, idx_pad]
                vb = v[b : b + 1, :, idx_pad]

                # Build a mask to null out padded tokens in attention
                # attn_bias: [1,H,Kmax,Kmax] with -inf where either row/col is padded
                if valid_len < Kmax:
                    pad_mask = torch.arange(Kmax, device=device) >= valid_len  # [Kmax]
                    # Row-wise (queries) padding: rows >= valid_len should NOT attend (we will drop them later).
                    # Column-wise (keys) padding: set scores to -inf for padded columns.
                    col_mask = pad_mask.view(1, 1, 1, Kmax)  # broadcast across rows
                    attn_bias = qb.new_full((1, H, Kmax, Kmax), 0.0)
                    attn_bias = attn_bias.masked_fill(col_mask, float("-inf"))
                else:
                    attn_bias = None

                # Causal mask based on **absolute** token order (not relative in the packed segment)
                # We compare original positions of idx_pad.
                if causal:
                    pos = abs_idx.index_select(0, idx_pad)  # [Kmax]
                    ipos = pos.view(1, 1, Kmax, 1)  # [1,1,Kmax,1]
                    jpos = pos.view(1, 1, 1, Kmax)  # [1,1,1,Kmax]
                    c_mask = jpos > ipos  # upper-triangular in absolute time
                    if attn_bias is None:
                        attn_bias = qb.new_zeros(1, H, Kmax, Kmax)
                    attn_bias = attn_bias.masked_fill(c_mask, float("-inf"))

                # Run attention kernel (Flash3/SDPA/RC). SDPA accepts an additive bias as "attn_mask".
                ob = self._attention_kernel(
                    qb, kb, vb, causal=False, attn_mask=attn_bias
                )  # [1,H,Kmax,Dh]

                # Scatter back only the valid part (ignore padded tail)
                if valid_len > 0:
                    out[b, :, idx_pad[:valid_len]] = ob[0, :, :valid_len]

        return out


"""Deprecated version (keep for reference)
class LSHv2AttentionCore(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_kv_heads: int,
                 bucket_size: int,
                 global_kind: str = "mamba",
                 prefer_flash3: bool = True,
                 rc_chunk_size: int = 2048,
                 compression_base: int = 4,
                 use_spectral: bool = True,
                 use_memory: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.bucket_size = bucket_size
        self.prefer_flash3 = prefer_flash3
        self.rc_chunk_size = rc_chunk_size

        self.global_enc = GlobalEncoder(d_model, kind=global_kind)
        self.gqa = GQALatent(d_model, n_heads, n_kv_heads)
        self.hasher = LearnedHasher(d_model, hash_dim=8, n_hashes=4)
        self.spectral = SpectralMixer(d_head=d_model // n_heads,
                                      rank=16, use_wavelet=use_spectral)
        self.mem = LTMemory(d_model, mem_slots=256, compress=True) if use_memory else None

        # per-head compression schedule: [1, s, s^2, s^3, ...]
        self.comp_factors = [max(1, compression_base ** (h // max(1, n_heads // 3))) for h in range(n_heads)]
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        \"""
        x: [B,N,D] → output: [B,N,D]
        \"""
        B, N, D = x.shape

        # 1) Global pre-encoding to make Q/K globally aware
        qg, kg = self.global_enc(x)                             # [B,N,D],[B,N,D]

        # 2) Project to multi-head Q/K/V (GQA)
        q, k, v = self.gqa(qg, kg, x)                           # [B,H,N,Dh] each (KV repeated if needed)

        # 3) Spectral low-rank + wavelet mix (optional)
        q, k = self.spectral(q, k)

        # 4) Learned hashing (two streams)
        sim_codes, ortho_codes = self.hasher(qg)                # [B,N,nh,Hd]
        bid_sim = LearnedHasher.to_bucket_ids(sim_codes, self.bucket_size, N)    # [B,N]
        bid_ort = LearnedHasher.to_bucket_ids(ortho_codes, self.bucket_size, N)  # [B,N]

        # 5) Read memory (if any) and concatenate as tiny prefix (KV only)
        if self.mem is not None:
            km, vm = self.mem.read(self.spectral.Ak(k).unsqueeze(0).squeeze(0))  # reuse proj dims; [B,H,M,Dc]
            # expand mem to same Dh by a cheap linear
            if km.numel() > 0:
                # project memory to head dim
                Dh = q.size(-1)
                up = nn.Linear(km.size(-1), Dh, bias=False).to(km.device, km.dtype)
                k_mem = up(km)
                v_mem = up(vm)
                # prepend memory along sequence dimension
                k = torch.cat([k_mem, k], dim=2)
                v = torch.cat([v_mem, v], dim=2)
                # pad bucket ids with a unique negative id (isolated bucket)
                pad_ids = bid_sim.new_full((B, k_mem.size(2)), -1)
                bid_sim = torch.cat([pad_ids, bid_sim], dim=1)
                bid_ort = torch.cat([pad_ids, bid_ort], dim=1)

        # 6) Per-head hierarchical compression + intra-bucket attention
        outs = []
        for h in range(self.n_heads):
            cf = max(1, int(self.comp_factors[h]))
            qh = compress_seq(q[:, h:h+1], cf)                  # [B,1,Nc,Dh]
            # map a KV head index under GQA
            kh = compress_seq(k[:, h:h+1], cf)
            vh = compress_seq(v[:, h:h+1], cf)
            Nc = qh.size(2)

            # resize bucket ids for this compressed length
            # choose stream: even heads use similarity, odd heads use orthogonality
            bids = bid_sim if (h % 2 == 0) else bid_ort
            bids_c = downsample_ids(bids, Nc)                   # [B,Nc]

            # intra-bucket attention (I/O-optimized)
            oh = intra_bucket_attn(qh, kh, vh,
                                   bucket_ids=bids_c,
                                   causal=causal,
                                   prefer_flash3=self.prefer_flash3,
                                   rc_chunk_size=self.rc_chunk_size,
                                   attn_mask=None)              # [B,1,Nc,Dh]

            # upsample back to N (token-aligned): nearest scatter
            if Nc != N:
                idx = torch.linspace(0, Nc - 1, N, device=x.device).round().long()  # nearest
                oh = oh.index_select(2, idx)
            outs.append(oh)

        o = torch.cat(outs, dim=1)                               # [B,H,N,Dh]
        o = o.transpose(1, 2).contiguous().view(B, N, D)         # merge heads
        o = self.out_proj(o)
        # write memory (lightweight summary)
        if self.mem is not None:
            with torch.no_grad():
                # compress raw k,v for writing
                k_comp = self.mem.key_proj(qg)
                v_comp = self.mem.val_proj(x)
                self.mem.write(k_comp.unsqueeze(1), v_comp.unsqueeze(1))
        return self.ln(o)
"""

# === Utilities: RMSNorm, covariance scheduler, pseudo-random eps ==================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class CovarianceScheduler(nn.Module):
    """
    Diagonal covariance scheduler for reparameterized sampling at layer outputs.
    sigma(depth) = softplus(diag_base) * (sigma0 * decay^(depth / (L-1)))
    """

    def __init__(self, dim: int, sigma0: float = 0.05, decay: float = 0.8):
        super().__init__()
        self.diag_base = nn.Parameter(torch.zeros(dim))  # learnable per-dim log-scale
        self.sigma0 = sigma0
        self.decay = decay

    def sigma_vec(self, depth: int, total_layers: int, device, dtype) -> torch.Tensor:
        if total_layers <= 1:
            scale = 1.0
        else:
            scale = self.sigma0 * (self.decay ** (depth / max(total_layers - 1, 1)))
        diag = F.softplus(self.diag_base).to(device=device, dtype=dtype)  # [D]
        return diag * scale  # [D]


# --- Deterministic per-call epsilon (legacy, not checkpoint-safe) ---
@torch.no_grad()
def _det_eps_like(ref: torch.Tensor, depth: int, tag: int = 0) -> torch.Tensor:
    """
    Deterministic N(0,1) noise with a per-call Generator seeded from a stable hash
    of the *reference input* tensor to the sublayer, the layer depth, and a tag.

    - ref: the input to f/g *before* the sublayer (e.g., pre-RMS(x2) or pre-RMS(y1))
    - depth: 0..L-1
    - tag: 0 for attention branch, 1 for MoE branch
    """
    # Build a seed from cheap, stable statistics and metadata.
    # (Using ref statistics keeps revnet invertibility: the same ref is recomputed in backward.)
    # Keep in 31-bit range for both CPU/CUDA generators.
    mean = ref.mean().item()
    std = ref.std().item()
    h = (
        int(abs(mean) * 1e6)
        ^ (int(abs(std) * 1e6) << 1)
        ^ (ref.shape[-1] << 11)
        ^ (ref.shape[-2] << 7)
        ^ (depth << 3)
        ^ (tag + 0x9E3779B1)
    ) & 0x7FFFFFFF
    if h == 0:
        h = 1

    # Create a device-appropriate generator and sample.
    if ref.is_cuda:
        g = torch.Generator(device=ref.device)
    else:
        g = torch.Generator()

    g.manual_seed(h)
    eps = torch.empty_like(ref)
    eps.normal_(mean=0.0, std=1.0, generator=g)  # works on CPU and CUDA
    return eps


# --- Stateless, checkpoint-safe epsilon ---
def _stateless_eps_like(x: torch.Tensor, depth: int, tag: int) -> torch.Tensor:
    """
    Produce deterministic N(0,1) noise with no dependence on global RNG state.
    Works on CPU/GPU; safe under gradient checkpoint recomputation.
    """
    # Derive a 64-bit seed from shape + (depth, tag)
    # Any stable hash is fine, but keep it purely integer and deterministic.
    s = x.shape
    h = (
        1469598103934665603
        ^ (s[-1] * 0x9E3779B185EBCA87)
        ^ (s[-2] * 0xC2B2AE3D27D4EB4F if x.dim() > 1 else 0)
        ^ (depth * 0xD6E8FEB86659FD93)
        ^ (tag * 0xA0761D6478BD642F)
    ) & 0xFFFFFFFF

    # Isolate RNG so global state is not affected:
    devices = [x.device] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(h))
        # No 'generator=' arg; keep compatibility with older PyTorch
        return torch.randn_like(x)


# === SwiGLU MoE (down-projection experts), top-k routing ========================


class SwiGLUDownExpert(nn.Module):
    """
    SwiGLU expert with down-projection rank r (r << D):
      u = x @ W_u  (D -> r)
      v = x @ W_v  (D -> r)
      s = silu(u) * v
      y = s @ W_o  (r -> D)
    """

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.W_u = nn.Linear(d_model, rank, bias=False)
        self.W_v = nn.Linear(d_model, rank, bias=False)
        self.W_o = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.W_u(x)
        v = self.W_v(x)
        s = F.silu(u) * v
        return self.W_o(s)


class TopKRouter(nn.Module):
    """
    Token-wise top-k router. Produces mixture weights per token over K experts.
    """

    def __init__(self, d_model: int, n_experts: int, k_active: int):
        super().__init__()
        self.n_experts = n_experts
        self.k_active = k_active
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [B, N, D]
        logits = self.gate(x)  # [B, N, K]
        topv, topi = torch.topk(logits, k=self.k_active, dim=-1)  # [B,N,k]
        # softmax within selected experts
        w = F.softmax(topv, dim=-1)  # [B,N,k]
        return topi, w  # indices of experts, and normalized weights


class SwiGLUMoEDownProj(nn.Module):
    """
    Mixture-of-Experts with down-projection SwiGLU experts.
    Uses token-wise top-k routing; combines expert outputs as weighted sum.
    """

    def __init__(
        self, d_model: int, n_experts: int = 8, k_active: int = 2, rank: int = None
    ):
        super().__init__()
        rank = rank or max(1, d_model // 4)  # default low rank: D/4
        self.router = TopKRouter(d_model, n_experts, k_active)
        self.experts = nn.ModuleList(
            [SwiGLUDownExpert(d_model, rank) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        topi, w = self.router(x)  # [B,N,k], [B,N,k]
        # Collect expert outputs per token in a loop over k (k is small).
        out = x.new_zeros(B, N, D)
        for j in range(w.size(-1)):
            idx = topi[..., j]  # [B,N]
            # Run each expert on the tokens routed to it. We do a simple gather-by-mask pass.
            # For efficiency and clarity, we compute all experts and gather per-token results.
            # (K small → acceptable)
            expert_outs = []
            for e_id, expert in enumerate(self.experts):
                y = expert(x)  # [B,N,D]
                expert_outs.append(y)
            # Stack and gather
            Y = torch.stack(expert_outs, dim=-2)  # [B,N,K,D]
            y_sel = torch.gather(
                Y, dim=-2, index=idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, D)
            ).squeeze(-2)  # [B,N,D]
            out = out + y_sel * w[..., j].unsqueeze(-1)
        return out


# === Reversible block with RMSNorm "sandwich" and reparameterized sampling ======
class RevBlock(nn.Module):
    """
    Reversible additive coupling:
        y1 = x1 + sample_f( f( pre_rms_f(x2) ), depth )
        y2 = x2 + sample_g( g( pre_rms_g(y1) ), depth )

    where sample_* uses reparameterization z = mean + sigma(depth) * eps(x, depth, tag),
    keeping reversibility since eps is a deterministic function of the available argument.

    f: attention sublayer (expects [B,N,Dh]) and returns mean in same shape.
    g: SwiGLU MoE sublayer (expects [B,N,Dh]) and returns mean in same shape.
    Dh = d_model // 2
    """

    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        d_half: int,
        sigma0_f: float = 0.05,
        sigma0_g: float = 0.05,
        decay_f: float = 0.8,
        decay_g: float = 0.8,
    ):
        super().__init__()
        self.f = f
        self.g = g
        self.pre_rms_f = RMSNorm(d_half)
        self.pre_rms_g = RMSNorm(d_half)
        self.cov_f = CovarianceScheduler(d_half, sigma0=sigma0_f, decay=decay_f)
        self.cov_g = CovarianceScheduler(d_half, sigma0=sigma0_g, decay=decay_g)

    # The sampler: deterministic per pass, but still random across passes
    def _sample(
        self,
        mean,
        *,
        sigma=None,
        ref=None,
        depth: int,
        total_layers: int,
        tag: int,
        rng_seed: int | None,
    ):
        D = mean.size(-1)
        if sigma is None:
            raise ValueError(
                "sigma must be provided or compute it before calling _sample."
            )
        # Compose a stable 64-bit seed from the pass seed + local coordinates.
        # This guarantees: same (rng_seed, depth, tag) => same eps; new rng_seed => new eps.
        key = (
            (0 if rng_seed is None else int(rng_seed))
            ^ (depth * 0x9E3779B185EBCA87)
            ^ (tag * 0xD1B54A32D192ED03)
        )
        key &= (1 << 63) - 1  # keep in 64-bit range

        g = torch.Generator(device=mean.device)
        g.manual_seed(key)  # explicit generator = no dependence on global RNG
        eps = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=g)

        return mean + eps * sigma.view(1, 1, D)

    def forward(
        self,
        x: torch.Tensor,
        depth: int,
        total_layers: int,
        rng_seed: int | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        Dh = D // 2
        x1, x2 = x[..., :Dh], x[..., Dh:]

        # Calculating Sigma with respect to the depth
        sigma_f = self.cov_f.sigma_vec(
            depth, total_layers, device=x.device, dtype=x.dtype
        )  # [Dh]
        sigma_g = self.cov_g.sigma_vec(
            depth, total_layers, device=x.device, dtype=x.dtype
        )  # [Dh]

        # Attention branch
        x2n = self.pre_rms_f(x2)
        m_f = self.f(x2n)
        z_f = self._sample(
            m_f,
            sigma=sigma_f,
            ref=x2n,
            depth=depth,
            total_layers=total_layers,
            tag=0,
            rng_seed=rng_seed,
        )
        y1 = x1 + z_f

        # MoE branch
        y1n = self.pre_rms_g(y1)
        m_g = self.g(y1n)
        z_g = self._sample(
            m_g,
            sigma=sigma_g,
            ref=y1n,
            depth=depth,
            total_layers=total_layers,
            tag=1,
            rng_seed=rng_seed,
        )
        y2 = x2 + z_g

        return torch.cat([y1, y2], dim=-1)


# === Full Block: uses attention core at Dh and SwiGLU-MoE at Dh ==================


class LSHv2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        bucket_size: int,
        global_kind: str,
        prefer_flash3: bool,
        rc_chunk_size: int,
        compression_base: int,
        use_spectral: bool,
        use_memory: bool,
        n_register: int = 4,
        n_mem_tokens: int = 4,
        reversible: bool = True,
        moe_experts: int = 8,
        moe_topk: int = 2,
        moe_rank: int = None,
        sigma0_f: float = 0.05,
        sigma0_g: float = 0.05,
        decay_f: float = 0.8,
        decay_g: float = 0.8,
    ):
        super().__init__()
        self.reversible = reversible
        Dh = (
            d_model // 2
        )  # attention & MoE work on half channels in reversible coupling

        attn = LSHv2AttentionCore(
            d_model=Dh,
            n_heads=max(1, n_heads // 2),  # keep head dim similar after halving D
            n_kv_heads=max(1, n_kv_heads // 2),
            bucket_size=bucket_size,
            global_kind=global_kind,
            prefer_flash3=prefer_flash3,
            rc_chunk_size=rc_chunk_size,
            compression_base=compression_base,
            use_spectral=use_spectral,
            use_memory=use_memory,
            n_register=n_register,
            n_mem_tokens=n_mem_tokens,
        )
        moe = SwiGLUMoEDownProj(
            d_model=Dh, n_experts=moe_experts, k_active=moe_topk, rank=moe_rank
        )

        if reversible:
            self.core = RevBlock(
                f=attn,
                g=moe,
                d_half=Dh,
                sigma0_f=sigma0_f,
                sigma0_g=sigma0_g,
                decay_f=decay_f,
                decay_g=decay_g,
            )
        else:
            # Non-reversible (kept for completeness)
            self.pre_rms_attn = RMSNorm(d_model)
            self.pre_rms_moe = RMSNorm(d_model)
            self.attn = attn
            self.moe = moe

    def forward(
        self,
        x: torch.Tensor,
        depth: int,
        total_layers: int,
        rng_seed: int | None = None,
    ) -> torch.Tensor:
        if self.reversible:
            return self.core(
                x, depth=depth, total_layers=total_layers, rng_seed=rng_seed
            )
        else:
            # Pre-norm → sublayer → residual (no sampling here if not reversible)
            h = x + self.attn(self.pre_rms_attn(x))
            h = h + self.moe(self.pre_rms_moe(h))
            return h


# === Model: weight tying / repeated layers enabled by default ===================


class LSHv2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        n_layers: int = 24,
        weight_tied_layers: int = 6,  # <-- tie repeats enabled by default
        max_seq_len: int = 131072,
        bucket_size: int = 128,
        global_kind: str = "mamba",
        prefer_flash3: bool = True,
        rc_chunk_size: int = 2048,
        compression_base: int = 4,
        use_spectral: bool = True,
        use_memory: bool = True,
        n_register: int = 4,
        n_mem_tokens: int = 4,
        reversible: bool = True,
        gradient_checkpoint: bool = False,
        moe_experts: int = 8,
        moe_topk: int = 2,
        moe_rank: int = None,
        sigma0_f: float = 0.05,
        sigma0_g: float = 0.05,
        decay_f: float = 0.8,
        decay_g: float = 0.8,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for reversible split"
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.gradient_checkpoint = gradient_checkpoint
        self.n_layers = n_layers
        self.tie = max(1, int(weight_tied_layers))

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)

        # Create a small bank of base blocks to be cycled (weight tying)
        self.base_blocks = nn.ModuleList(
            [
                LSHv2Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    bucket_size=bucket_size,
                    global_kind=global_kind,
                    prefer_flash3=prefer_flash3,
                    rc_chunk_size=rc_chunk_size,
                    compression_base=compression_base,
                    use_spectral=use_spectral,
                    use_memory=use_memory,
                    n_register=n_register,
                    n_mem_tokens=n_mem_tokens,
                    reversible=reversible,
                    moe_experts=moe_experts,
                    moe_topk=moe_topk,
                    moe_rank=moe_rank,
                    sigma0_f=sigma0_f,
                    sigma0_g=sigma0_g,
                    decay_f=decay_f,
                    decay_g=decay_g,
                )
                for _ in range(self.tie)
            ]
        )

        # Final RMSNorm + tied LM head
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight  # tie

        self.apply(self._init_weights)
        print(
            f"[LSHv2] params: {sum(p.numel() for p in self.parameters()):,} | "
            f"flash3={'yes' if _HAS_FA3 else 'no'} | reversible={reversible} | "
            f"tied_groups={self.tie} over {self.n_layers} layers"
        )

        # Output Size Summary
        sizes = self.component_sizes()
        print(
            "  component sizes (approx): "
            + ", ".join([f"{k}={v}" for k, v in sizes.items()])
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, input_ids: torch.Tensor, rng_seed: int | None = None, causal: bool = True
    ) -> Dict:
        B, N = input_ids.shape
        if N > self.max_seq_len:
            raise ValueError(f"seq_len {N} > max_seq_len {self.max_seq_len}")

        pos = torch.arange(N, device=input_ids.device)
        x = self.emb(input_ids) + self.pos(pos)[None, :, :]

        # Cycle through tied blocks; pass depth to support covariance scheduling & reparam
        for depth in range(self.n_layers):
            blk = self.base_blocks[depth % self.tie]
            if self.gradient_checkpoint and self.training:
                # capture depth in lambda; pass total_layers
                x = checkpoint(
                    lambda y, d=depth, rs=rng_seed: blk(
                        y, depth=d, total_layers=self.n_layers, rng_seed=rs
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = blk(x, depth=depth, total_layers=self.n_layers)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return {"logits": logits, "last_hidden_state": x}

    # Convenience: model size in approximate number of parameters
    def size_str(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        if n_params < 1e6:
            return f"{n_params / 1e3:.1f}K"
        elif n_params < 1e9:
            return f"{n_params / 1e6:.1f}M"
        else:
            return f"{n_params / 1e9:.1f}B"

    # Size of each component in parameters
    def component_sizes(self) -> Dict[str, str]:
        sizes = {}
        sizes["embeddings"] = (
            f"{sum(p.numel() for p in self.emb.parameters()) / 1e6:.1f}M"
        )
        sizes["positional"] = (
            f"{sum(p.numel() for p in self.pos.parameters()) / 1e3:.1f}K"
        )
        sizes["blocks"] = (
            f"{sum(p.numel() for p in self.base_blocks.parameters()) / 1e6:.1f}M"
        )
        sizes["ln_f"] = f"{sum(p.numel() for p in self.ln_f.parameters()) / 1e3:.1f}K"
        sizes["lm_head"] = (
            f"{sum(p.numel() for p in self.lm_head.parameters()) / 1e6:.1f}M"
        )
        return sizes


# ---------- Factory ----------
def create_lsh_v2_model(
    vocab_size: int = 32000, target_params: str = "100M"
) -> LSHv2Model:
    cfg = dict(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=16,
        n_kv_heads=4,
        n_layers=24,
        weight_tied_layers=1,  # default: weight tying enabled
        max_seq_len=131072,
        bucket_size=2048,
        global_kind="mamba",
        prefer_flash3=True,
        rc_chunk_size=2048,
        compression_base=4,
        use_spectral=True,
        use_memory=True,
        n_register=4,
        n_mem_tokens=4,  # registers & memory tokens on by default
        reversible=True,
        gradient_checkpoint=False,
        moe_experts=2,
        moe_topk=2,
        moe_rank=None,
        sigma0_f=0.05,
        sigma0_g=0.05,
        decay_f=0.8,
        decay_g=0.8,
    )
    model = LSHv2Model(**cfg)

    # iterate over whichever attribute exists
    blocks = getattr(model, "base_blocks", None) or getattr(model, "blocks", None)
    for blk in blocks:
        # reversible path exposes attention at blk.core.f; non-reversible uses blk.attn
        attn = (
            getattr(blk.core, "f", None)
            if hasattr(blk, "core")
            else getattr(blk, "attn", None)
        )
        if attn is None:
            continue
        # ensure these attributes match your desired defaults
        if hasattr(attn, "n_register"):
            attn.n_register = cfg["n_register"]
        if hasattr(attn, "n_mem_tokens"):
            attn.n_mem_tokens = cfg["n_mem_tokens"]

    return model


if __name__ == "__main__":
    # Minimal smoke test
    torch.manual_seed(0)
    m = create_lsh_v2_model(32000, "100M")
    x = torch.randint(0, 32000, (1, 1024))
    with torch.no_grad():
        y = m(x, causal=True)
    print("OK:", y["logits"].shape)
