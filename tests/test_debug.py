import torch
import pytest
import numpy as np
from typing import List, Optional
import flash_attn_wrapper  # noqa: F401
import sys

def ref_paged_attn(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_lens: List[int],
        kv_lens: List[int],
        block_tables: torch.Tensor,
        scale: float,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                                      (query_len + sliding_window) +
                                                      1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)

def test_flash_attn_kvcache():

    # 配置参数
    block_size = 32
    num_heads = (8, 2)
    head_size = 256
    num_blocks =2048

    kv_lens = [1328, 18, 463]
    soft_cap = None

    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)

    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    results = torch.ops.vllm.flash_attn_with_kvcache(
        decode_query=query.unsqueeze(1),
        key_cache=key_cache,
        value_cache=value_cache,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
        cache_seqlens=kv_lens_tensor,
        softcap=soft_cap if soft_cap is not None else 0,
        return_softmax_lse=False,
    )

    print(f"results: {results.shape}")
    output = results.squeeze(1)
    if num_blocks <= 2048:
        test_utils = ["test_faketensor", "test_schema"]
    else:
        test_utils = ["test_faketensor"]

    torch.library.opcheck(torch.ops.vllm.flash_attn_with_kvcache,
                          args=tuple(),
                          kwargs=dict(
                              decode_query=query.unsqueeze(1),
                              key_cache=key_cache,
                              value_cache=value_cache,
                              softmax_scale=scale,
                              causal=True,
                              block_table=block_tables,
                              cache_seqlens=kv_lens_tensor,
                              softcap=soft_cap if soft_cap is not None else 0,
                          ),
                          test_utils=test_utils)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )


    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"
        

if __name__ == "__main__":
    test_flash_attn_kvcache()
