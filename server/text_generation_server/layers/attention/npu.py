import torch
import torch_npu  # noqa: F401
from typing import Optional
from text_generation_server.models.flash_causal_lm import BLOCK_SIZE
from text_generation_server.layers.attention import Seqlen

SUPPORTS_WINDOWING = False


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
):
    num_heads, head_size = key.shape
    num_blocks, block_size = key_cache.shape[:2]
    total_blocks = num_blocks * block_size

    # only support contiguous k, v
    key = key.contiguous()
    value = value.contiguous()

    key_cache_reshaped = key_cache.view(total_blocks, num_heads, head_size)
    value_cache_reshaped = value_cache.view(total_blocks, num_heads, head_size)

    torch_npu.npu_scatter_nd_update_(key_cache_reshaped, slots, key)
    torch_npu.npu_scatter_nd_update_(value_cache_reshaped, slots, value)


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    seqlen: Seqlen,
    max_s: int,
    softcap: Optional[float] = None,
):
    # value shape: [num_tokens, num_heads, head_size]
    # value_cache => [num_blocks, block_size, num_heads*head_size]
    block_size = BLOCK_SIZE
    num_heads = query.shape[1]

    breakpoint()
    return torch_npu.npu_incre_flash_attention(
        query,
        key_cache,
        value_cache,
        num_heads=num_heads,
        num_key_value_heads=kv_head_mapping,
        scale_value=softmax_scale,
        input_layout="BSH",
        block_table=block_tables,
        block_size=block_size,
        actual_seq_length=max_s,
    )


def attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlen: Seqlen,
    block_tables: torch.Tensor,
    softmax_scale,
    window_size_left=-1,
    causal=True,
    softcap: Optional[float] = None,
):
    out = torch.empty_like(q)

    # We do not need to check window_size_left (not supported) here, so it is already checked ahead of time at model load.
    head_num = q.shape[-2]

    cu_seqlens_q = seqlen.cu_seqlen_q[1:].tolist()
    cu_seqlens_k = seqlen.cu_seqlen_k[1:].tolist()
    seqlen_q = min(seqlen.max_q, 2048)
    seqlen_k = min(seqlen.max_k, 2048)

    if seqlen.max_q < 2048:
        sparse_mode = 0
    else:
        sparse_mode = 2

    attn_mask = (
        torch.triu(
            torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    return torch_npu.npu_fusion_attention(
        q,
        key_cache,
        value_cache,
        head_num,
        "TND",
        atten_mask=attn_mask,
        scale=softmax_scale,
        pre_tockens=q.shape[0],
        next_tockens=0,
        keep_prob=1,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,
    )[0]
