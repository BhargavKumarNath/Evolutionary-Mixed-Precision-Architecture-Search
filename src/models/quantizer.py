import torch


def fake_quantize_tensor(w: torch.Tensor, n_bits: int, group_size: int = 128) -> torch.Tensor:
    """Simulates quantisation effects on a weight tensor
    
    Args:
        w: Weight tensor (Output Features, Input Features)
        n_bits: Target bit-width (2, 4, 8)
        group_size: Granularity of quantisation (simulates hardware blocks)
    
    Returns:
        De-quantised weights (simulating the loss of precision)"""
    if n_bits >= 16:
        return w
    
    # 1. Reshape to groups or finer granularity
    orig_shape = w.shape
    w_flat = w.reshape(-1, group_size)

    # 2. Calculate scale (AbsMax)
    max_val = w_flat.abs().max(dim=1, keepdim=True)[0]
    max_val = max_val.clamp(min=1e-5) # Avoid div by zero

    limit = 2 ** (n_bits - 1) - 1
    scale = max_val / limit

    # 3. Quantize
    w_quant = (w_flat / scale).round().clamp(-limit, limit)

    # 4. Dequantize
    w_dequant = w_quant * scale

    return w_dequant.reshape(orig_shape)
