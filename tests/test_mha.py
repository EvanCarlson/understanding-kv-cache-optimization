"""Sanity check for TinyMultiHeadAttention's math, independent of any KV
cache: given identical weights, our from-scratch causal MHA forward pass
must match PyTorch's built-in nn.MultiheadAttention. If these diverge, the
attention math itself is wrong before a cache ever enters the picture.
"""

import torch

from model import ModelConfig, TinyMultiHeadAttention


def test_mha_matches_torch_mha() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        max_seq_len=16,
        n_layers=2,
        num_heads=4,
        num_kv_heads=4,
    )
    mha = TinyMultiHeadAttention(config=config, layer_idx=0)
    mha.eval()

    # Random input sequence of length 6
    input_seq = torch.randn(1, 6, config.d_model)

    # Compute output using our TinyMultiHeadAttention implementation
    our_output = mha(input_seq)

    # Compute output using PyTorch's built-in MultiheadAttention
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=config.d_model,
        num_heads=config.num_heads,
        batch_first=True,
        add_bias_kv=False,
        bias=True,
    )
    torch_mha.eval()

    # Copy our weights (and biases, now that both projections use them) over
    # so both implementations compute the same function — otherwise we'd
    # just be comparing two random models.
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(
            torch.cat([mha.Q_proj.weight, mha.K_proj.weight, mha.V_proj.weight], dim=0)
        )
        torch_mha.in_proj_bias.copy_(
            torch.cat([mha.Q_proj.bias, mha.K_proj.bias, mha.V_proj.bias], dim=0)
        )
        torch_mha.out_proj.weight.copy_(mha.out_proj.weight)
        torch_mha.out_proj.bias.copy_(mha.out_proj.bias)

    # Our implementation always applies a causal mask, so give torch's
    # implementation the same mask -- otherwise it defaults to full
    # (non-causal) attention and the outputs can't match.
    seq_len = input_seq.shape[1]
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    torch_output, _ = torch_mha(input_seq, input_seq, input_seq, attn_mask=causal_mask)

    # Check that the outputs are close enough
    assert torch.allclose(our_output, torch_output, atol=1e-4)
