import torch

# 通过 torch.ops 直接访问，避免导入问题
print("测试 C++ 扩展返回值:")

# 设置设备
torch.set_default_device("cuda")
torch.cuda.manual_seed_all(0)

# 创建简单的测试数据
batch_size = 1
seqlen_q = 1
num_heads = 2
head_size = 64
num_blocks = 4
block_size = 32

q = torch.randn(batch_size, seqlen_q, num_heads, head_size, dtype=torch.float16)
k_cache = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.float16)
v_cache = torch.randn_like(k_cache)
cache_seqlens = torch.tensor([10], dtype=torch.int32)
block_table = torch.tensor([[0, 1]], dtype=torch.int32)

# 首先检查操作符是否存在
try:
    # 先导入 vllm_flash_attn_c，这样操作符才能注册
    print("尝试导入模块...")
    import vllm_flash_attn_c
    print("模块导入成功")
except Exception as e:
    print(f"模块导入失败: {e}")
    try:
        # 尝试通过其他方式
        from vllm_flash_attn import vllm_flash_attn_c
        print("通过 vllm_flash_attn 导入成功")
    except Exception as e2:
        print(f"替代导入也失败: {e2}")

# 检查操作符是否可用
print(f"可用的 vllm_flash_attn_c 操作: {dir(torch.ops.vllm_flash_attn_c) if hasattr(torch.ops, 'vllm_flash_attn_c') else '未找到'}")

if hasattr(torch.ops, 'vllm_flash_attn_c') and hasattr(torch.ops.vllm_flash_attn_c, 'fwd_kvcache'):
    print("直接调用 C++ 扩展:")
    try:
        result = torch.ops.vllm_flash_attn_c.fwd_kvcache(
            q,
            k_cache,
            v_cache,
            None,  # k
            None,  # v
            cache_seqlens,
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # cache_batch_idx
            block_table,
            None,  # alibi_slopes
            None,  # out
            (head_size ** -0.5),  # softmax_scale
            True,  # is_causal
            -1,    # window_size_left
            -1,    # window_size_right
            0.0,   # softcap
            True,  # is_rotary_interleaved
            0      # num_splits
        )
        
        print(f"返回值类型: {type(result)}")
        if isinstance(result, (list, tuple)):
            print(f"返回值数量: {len(result)}")
            for i, item in enumerate(result):
                print(f"result[{i}]: {type(item)} - {item.shape if hasattr(item, 'shape') else item}")
        else:
            print(f"返回单个值: {result.shape}")
            
    except Exception as e:
        print(f"调用错误: {e}")
else:
    print("fwd_kvcache 操作符不可用") 