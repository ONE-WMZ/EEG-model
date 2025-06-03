
def block_split(tensor, chunk_size=64):
    # 分块后的张量，形状 (B, C, N, chunk_size) = (8, 22, 8, 64)
    B, C, L = tensor.shape
    assert L % chunk_size == 0, f"序列长度{L}必须能被{chunk_size}整除"
    # 分块操作 (两步reshape保证内存连续)
    chunks = tensor.reshape(B, C, -1, chunk_size)  # (8,22,8,64)
    return chunks.contiguous()