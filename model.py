from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar, cast, runtime_checkable

import torch
from torch import nn
from torch.profiler import record_function


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    max_seq_len: int
    n_layers: int
    num_heads: int
    # Specific to GQA. num_kv_heads = 0 -> MHA
    num_kv_heads: int


@dataclass(frozen=True)
class KVCacheAppendResult:
    keys: torch.Tensor
    values: torch.Tensor


@runtime_checkable
class KVCacheProtocol(Protocol):
    def append(
        self,
        *,
        layer_idx: int,
        keys: torch.Tensor,  # Shape: [B, H, S, K]
        values: torch.Tensor,  # Shape: [B, H, S, K]
    ) -> KVCacheAppendResult:
        """Write new key/value tensors into the cache at the given layer index."""

    def advance(self, *, num_new_tokens: int) -> None:
        """Commit the step's writes, advancing the cache's shared sequence length.

        Every layer must call `append()` for the current step before this is
        called. `append()` writes are only valid once every layer has
        written; calling `advance()` before all layers have appended, or
        appending twice for one step, desynchronizes the per-layer views of
        the cache's length.
        """

    @property
    def length(self) -> int:
        """Committed sequence length represented by every decoder layer's cache."""

    @property
    def used_nbytes(self) -> int:
        """Bytes representing live cached tokens, given the cache's current length."""

    @property
    def allocated_nbytes(self) -> int:
        """Bytes physically reserved by the cache right now, used or not."""


T = TypeVar("T", bound=KVCacheProtocol)


class KVCacheFactory:
    _registry: ClassVar[dict[str, type[KVCacheProtocol]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(subclass: type[T]) -> type[T]:
            if not isinstance(subclass, KVCacheProtocol):
                raise TypeError(
                    f"Class {subclass.__name__} does not match KVCacheProtocol"
                )

            normalized_name = name.lower()
            if normalized_name in cls._registry:
                raise ValueError(f"{name} is already registered")

            cls._registry[normalized_name] = subclass

            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> KVCacheProtocol:
        creator_class = cls._registry.get(name.lower())
        if not creator_class:
            raise ValueError(f"Unknown KV cache: {name}")

        return creator_class(*args, **kwargs)


def _compute_head_dim(*, d_model: int, num_heads: int) -> int:
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

    return d_model // num_heads


def _validate_append_inputs(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    layer_idx: int,
    n_layers: int,
    expected_batch_size: int,
    expected_num_heads: int,
    expected_head_dim: int,
) -> None:
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError("keys and values must both have shape [B, H, S, K]")

    if keys.shape != values.shape:
        raise ValueError(
            f"keys and values must have identical shapes; "
            f"got {tuple(keys.shape)} and {tuple(values.shape)}"
        )

    if not 0 <= layer_idx < n_layers:
        raise IndexError(f"layer_idx={layer_idx} is out of range for {n_layers} layers")

    batch_size, num_heads, _, head_dim = keys.shape

    if batch_size != expected_batch_size:
        raise ValueError(f"expected batch_size={expected_batch_size}, got {batch_size}")

    if num_heads != expected_num_heads:
        raise ValueError(f"expected num_heads={expected_num_heads}, got {num_heads}")

    if head_dim != expected_head_dim:
        raise ValueError(f"expected head_dim={expected_head_dim}, got {head_dim}")


class BaseKVCache:
    def __init__(
        self,
        *,
        max_seq_len: int,
    ) -> None:
        self.max_seq_len = max_seq_len
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def advance(self, *, num_new_tokens: int) -> None:
        self._length += num_new_tokens

        if self._length > self.max_seq_len:
            raise RuntimeError(
                f"KV cache length exceeded max_seq_len: length={self._length}"
            )

    def _bounds_for_append(self, seq_len: int) -> tuple[int, int]:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        start = self._length
        end = start + seq_len

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: requested end={end}, max_seq_len={self.max_seq_len}"
            )

        return start, end


class PreallocatedKVCache(BaseKVCache):
    def __init__(
        self,
        *,
        n_layers: int,
        num_cache_heads: int,
        head_dim: int,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(max_seq_len=max_seq_len)

        self.n_layers = n_layers
        self.head_dim = head_dim

        shape = (n_layers, batch_size, num_cache_heads, max_seq_len, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def used_nbytes(self) -> int:
        _, batch_size, num_cache_heads, _, head_dim = self.keys.shape
        elems_per_tensor = (
            self.n_layers * batch_size * num_cache_heads * self._length * head_dim
        )

        return 2 * elems_per_tensor * self.keys.element_size()

    @property
    def allocated_nbytes(self) -> int:
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()


# Store K/V exactly as computed. Simple, but memory-heavy.
@KVCacheFactory.register("basic")
class BasicKVCache(PreallocatedKVCache):
    def __init__(
        self,
        *,
        config: ModelConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.num_heads = config.num_heads
        super().__init__(
            n_layers=config.n_layers,
            num_cache_heads=config.num_heads,
            head_dim=_compute_head_dim(
                d_model=config.d_model, num_heads=config.num_heads
            ),
            batch_size=batch_size,
            max_seq_len=config.max_seq_len,
            dtype=torch.float32,
            device=device,
        )

    def append(
        self,
        *,
        layer_idx: int,
        keys: torch.Tensor,  # Shape: [B, H, S, K]
        values: torch.Tensor,  # Shape: [B, H, S, K]
    ) -> KVCacheAppendResult:
        _validate_append_inputs(
            keys=keys,
            values=values,
            layer_idx=layer_idx,
            n_layers=self.n_layers,
            expected_batch_size=self.keys.shape[1],
            expected_num_heads=self.num_heads,
            expected_head_dim=self.head_dim,
        )

        seq_len = keys.shape[2]
        start, end = self._bounds_for_append(seq_len)

        with record_function("cache_write"):
            self.keys[layer_idx, :, :, start:end, :] = keys
            self.values[layer_idx, :, :, start:end, :] = values

        with record_function("cache_read"):
            all_keys = self.keys[layer_idx, :, :, :end, :]
            all_values = self.values[layer_idx, :, :, :end, :]

        return KVCacheAppendResult(keys=all_keys, values=all_values)


# Reduce how many heads that are stored.
@KVCacheFactory.register("gqa")
class GQAKVCache(PreallocatedKVCache):
    def __init__(
        self,
        *,
        config: ModelConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        num_kv_heads = config.num_kv_heads

        if not num_kv_heads:
            raise ValueError(f"expected num_kv_heads > 0, got {num_kv_heads}")

        self.num_kv_heads = num_kv_heads

        super().__init__(
            n_layers=config.n_layers,
            num_cache_heads=num_kv_heads,
            head_dim=_compute_head_dim(
                d_model=config.d_model, num_heads=config.num_heads
            ),
            batch_size=batch_size,
            max_seq_len=config.max_seq_len,
            dtype=torch.float32,
            device=device,
        )

    def append(
        self,
        *,
        layer_idx: int,
        keys: torch.Tensor,  # Shape: [B, num_kv_heads, S, K]
        values: torch.Tensor,  # Shape: [B, num_kv_heads, S, K]
    ) -> KVCacheAppendResult:
        _validate_append_inputs(
            keys=keys,
            values=values,
            layer_idx=layer_idx,
            n_layers=self.n_layers,
            expected_batch_size=self.keys.shape[1],
            expected_num_heads=self.num_kv_heads,
            expected_head_dim=self.head_dim,
        )

        seq_len = keys.shape[2]
        start, end = self._bounds_for_append(seq_len)

        with record_function("cache_write"):
            self.keys[layer_idx, :, :, start:end, :] = keys
            self.values[layer_idx, :, :, start:end, :] = values

        with record_function("cache_read"):
            all_keys = self.keys[layer_idx, :, :, :end, :]
            all_values = self.values[layer_idx, :, :, :end, :]

        return KVCacheAppendResult(keys=all_keys, values=all_values)


# Store tokens in fixed blocks, allocated on demand instead of one
# worst-case-sized reservation per sequence.
#
# This demonstrates block-based allocation, not a fused PagedAttention
# kernel: attention still reads a full contiguous K/V tensor, concatenated
# from blocks on every append below, so it doesn't get PagedAttention's
# non-contiguous page-table read path or its shared physical block pool
# across sequences.
@KVCacheFactory.register("paged")
class BlockedKVCache(BaseKVCache):
    def __init__(
        self,
        *,
        config: ModelConfig,
        batch_size: int,
        device: torch.device,
        block_size: int = 128,
    ) -> None:
        super().__init__(max_seq_len=config.max_seq_len)

        self.n_layers = config.n_layers
        self.head_dim = _compute_head_dim(
            d_model=config.d_model, num_heads=config.num_heads
        )
        self.num_heads = config.num_heads
        self.batch_size = batch_size
        self.block_size = block_size
        self.dtype = torch.float32
        self.device = device

        # One independent block list per layer.
        self.layer_blocks: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(config.n_layers)
        ]

    @property
    def used_nbytes(self) -> int:
        element_size = torch.zeros((), dtype=self.dtype).element_size()
        elems_per_tensor = (
            self.batch_size * self.num_heads * self._length * self.head_dim
        )

        return 2 * elems_per_tensor * element_size

    @property
    def allocated_nbytes(self) -> int:
        total_blocks = sum(len(blocks) for blocks in self.layer_blocks)

        if total_blocks == 0:
            return 0

        element_size = torch.zeros((), dtype=self.dtype).element_size()
        bytes_per_block = (
            self.batch_size * self.num_heads * self.block_size * self.head_dim
        ) * element_size

        return 2 * total_blocks * bytes_per_block

    def append(
        self,
        *,
        layer_idx: int,
        keys: torch.Tensor,  # Shape: [B, H, S, K]
        values: torch.Tensor,  # Shape: [B, H, S, K]
    ) -> KVCacheAppendResult:
        _validate_append_inputs(
            keys=keys,
            values=values,
            layer_idx=layer_idx,
            n_layers=self.n_layers,
            expected_batch_size=self.batch_size,
            expected_num_heads=self.num_heads,
            expected_head_dim=self.head_dim,
        )

        batch_size, _, seq_len, head_dim = keys.shape
        start, end = self._bounds_for_append(seq_len)
        blocks = self.layer_blocks[layer_idx]

        # Write the new chunk into blocks of size block_size.
        # e.g. block_size=128, seq_len=300 -> 3 blocks of size (128, 128, 44).
        with record_function("cache_write"):
            pos = start
            written = 0

            while written < seq_len:
                block_idx = pos // self.block_size
                offset_in_block = pos % self.block_size

                if block_idx == len(blocks):
                    new_keys_block = torch.zeros(
                        (batch_size, self.num_heads, self.block_size, head_dim),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    new_values_block = torch.zeros_like(new_keys_block)
                    blocks.append((new_keys_block, new_values_block))

                k_block, v_block = blocks[block_idx]
                write_now = min(self.block_size - offset_in_block, seq_len - written)

                src_slice = slice(written, written + write_now)
                dst_slice = slice(offset_in_block, offset_in_block + write_now)

                k_block[:, :, dst_slice, :] = keys[:, :, src_slice, :]
                v_block[:, :, dst_slice, :] = values[:, :, src_slice, :]

                pos += write_now
                written += write_now

        with record_function("cache_read"):
            # (B, H, block_size, d) * num_blocks → (B, H, total_tokens, d)
            all_keys = torch.cat([kb for kb, _ in blocks], dim=2)[:, :, :end, :]
            all_values = torch.cat([vb for _, vb in blocks], dim=2)[:, :, :end, :]

        return KVCacheAppendResult(keys=all_keys, values=all_values)


# Asymmetric uint8 affine quantization. Each append() call quantizes its own
# chunk with one scale/zero-point, computed from that chunk's min/max. Those
# parameters are stored per cached position (broadcast across the chunk that
# produced them), not per layer, so a later append's scale/zero-point never
# overwrites the parameters an earlier chunk's tokens were quantized with.
@KVCacheFactory.register("quantized")
class QuantizedKVCache(PreallocatedKVCache):
    def __init__(
        self,
        *,
        config: ModelConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.num_heads = config.num_heads

        super().__init__(
            n_layers=config.n_layers,
            num_cache_heads=config.num_heads,
            head_dim=_compute_head_dim(
                d_model=config.d_model, num_heads=config.num_heads
            ),
            batch_size=batch_size,
            max_seq_len=config.max_seq_len,
            dtype=torch.uint8,
            device=device,
        )

        # Same [n_layers, B, H, S, 1] shape as self.keys/self.values, but
        # with the head_dim axis collapsed to 1: one scale/zero-point per
        # cached position, not per element.
        metadata_shape = (*self.keys.shape[:-1], 1)
        self.k_scale = torch.ones(metadata_shape, device=device)
        self.k_zero = torch.zeros(metadata_shape, device=device)

        self.v_scale = torch.ones(metadata_shape, device=device)
        self.v_zero = torch.zeros(metadata_shape, device=device)

    @property
    def used_nbytes(self) -> int:
        _, batch_size, num_cache_heads, _, _ = self.keys.shape
        used_metadata_elems = (
            self.n_layers * batch_size * num_cache_heads * self._length
        )

        return (
            super().used_nbytes + 4 * used_metadata_elems * self.k_scale.element_size()
        )

    @property
    def allocated_nbytes(self) -> int:
        scale_tensors = (self.k_scale, self.k_zero, self.v_scale, self.v_zero)
        scale_bytes = sum(t.numel() * t.element_size() for t in scale_tensors)

        return super().allocated_nbytes + scale_bytes

    def _quantize_tensor(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_min = x.min()
        x_max = x.max()

        eps = 1e-6
        scale = (x_max - x_min) / 255.0
        scale = torch.clamp(scale, min=eps)

        zero_point = x_min

        x_q = ((x - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)

        return x_q, scale, zero_point

    def _dequantize_tensor(
        self,
        x_q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        return x_q.float() * scale + zero_point

    def append(
        self,
        *,
        layer_idx: int,
        keys: torch.Tensor,  # Shape: [B, H, S, K]
        values: torch.Tensor,  # Shape: [B, H, S, K]
    ) -> KVCacheAppendResult:
        _validate_append_inputs(
            keys=keys,
            values=values,
            layer_idx=layer_idx,
            n_layers=self.n_layers,
            expected_batch_size=self.keys.shape[1],
            expected_num_heads=self.num_heads,
            expected_head_dim=self.head_dim,
        )

        seq_len = keys.shape[2]
        start, end = self._bounds_for_append(seq_len)

        with record_function("cache_write"):
            k_q, k_scale, k_zero = self._quantize_tensor(keys)
            v_q, v_scale, v_zero = self._quantize_tensor(values)

            self.keys[layer_idx, :, :, start:end, :] = k_q
            self.values[layer_idx, :, :, start:end, :] = v_q

            # k_scale/k_zero are scalars for this chunk; broadcasting them
            # into the chunk's own slice keeps earlier positions' stored
            # parameters untouched.
            self.k_scale[layer_idx, :, :, start:end, :] = k_scale
            self.k_zero[layer_idx, :, :, start:end, :] = k_zero
            self.v_scale[layer_idx, :, :, start:end, :] = v_scale
            self.v_zero[layer_idx, :, :, start:end, :] = v_zero

        with record_function("cache_read"):
            q_k_all = self.keys[layer_idx, :, :, :end, :]
            q_v_all = self.values[layer_idx, :, :, :end, :]

            k_scale_all = self.k_scale[layer_idx, :, :, :end, :]
            k_zero_all = self.k_zero[layer_idx, :, :, :end, :]
            v_scale_all = self.v_scale[layer_idx, :, :, :end, :]
            v_zero_all = self.v_zero[layer_idx, :, :, :end, :]

            all_keys = self._dequantize_tensor(q_k_all, k_scale_all, k_zero_all)
            all_values = self._dequantize_tensor(q_v_all, v_scale_all, v_zero_all)

        return KVCacheAppendResult(keys=all_keys, values=all_values)


class TinyMultiHeadAttention(nn.Module):
    def __init__(self, *, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()

        num_query_heads = config.num_heads
        num_kv_heads = config.num_kv_heads or config.num_heads

        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.d_model = config.d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = config.d_model // num_query_heads
        self.layer_idx = layer_idx

        # We always use a full-width query projection, regardless of whether
        # we use MHA (num_heads key/value heads), MQA (1 key/value head), or
        # GQA (1 < key/value heads < num_heads).
        self.Q_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        kv_dim = self.num_kv_heads * self.head_dim
        self.K_proj = nn.Linear(config.d_model, kv_dim, bias=True)
        self.V_proj = nn.Linear(config.d_model, kv_dim, bias=True)

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        # A buffer is automatically moved to the correct device whenever the
        # model itself is (model.to(device)), so the causal mask never needs
        # to be moved by hand inside forward(). This prevents host-to-device
        # syncs from model.to(device) calls.
        self.mask: torch.Tensor
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        )

        # Maps each query head to its KV head
        # e.g. num_query_heads=8, num_kv_heads=2 -> [0,0,0,0,1,1,1,1]
        self.kv_group_idx: torch.Tensor
        group_size = self.num_query_heads // self.num_kv_heads
        self.register_buffer(
            "kv_group_idx", torch.arange(self.num_query_heads) // group_size
        )

    def forward(
        self,
        x: torch.Tensor,  # Shape: [B, S, D]
        *,
        cache: KVCacheProtocol | None = None,
    ) -> torch.Tensor:
        with record_function("qkv_projection"):
            queries = self.Q_proj(x)
            keys = self.K_proj(x)
            values = self.V_proj(x)

        # Split K, Q, V into multiple heads: [B, S, D] -> [B, S, H, K]
        batch_size, seq_len, _ = queries.shape

        queries = queries.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # [B, S, H, K] -> [B, H, S, K]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if cache is not None:
            append_result = cache.append(
                layer_idx=self.layer_idx,
                keys=keys,
                values=values,
            )
            all_keys, all_values = append_result.keys, append_result.values

            # Retrieve KV cache per head
            expanded_keys = all_keys[:, self.kv_group_idx, :, :]
            expanded_values = all_values[:, self.kv_group_idx, :, :]

            with record_function("attention_matmul"):
                # Scaled dot-product attention
                attn_scores = (
                    torch.matmul(queries, expanded_keys.transpose(2, 3))
                    / self.head_dim**0.5
                )

                # A single query (i.e. seq_len == 1) can see all past keys and values
                if seq_len > 1:
                    # A causal mask is only needed over the chunk's own final
                    # seq_len key columns, to stop a query at local position i
                    # from seeing chunk-local positions in the future.
                    local_mask = self.mask[:seq_len, :seq_len]
                    attn_scores[..., -seq_len:] = attn_scores[
                        ..., -seq_len:
                    ].masked_fill(
                        local_mask == 0,
                        float("-inf"),
                    )

                # Normalize attention scores
                attn_weights = torch.softmax(attn_scores, dim=-1)

                context_vect = torch.matmul(attn_weights, expanded_values)
        else:
            expanded_keys = keys[:, self.kv_group_idx, :, :]
            expanded_values = values[:, self.kv_group_idx, :, :]

            with record_function("attention_matmul"):
                attn_scores = (
                    torch.matmul(queries, expanded_keys.transpose(2, 3))
                    / self.head_dim**0.5
                )

                # Mask future positions (causal attention)
                attn_scores = attn_scores.masked_fill(
                    self.mask[:seq_len, :seq_len] == 0, -torch.inf
                )

                attn_weights = torch.softmax(attn_scores, dim=-1)

                context_vect = torch.matmul(attn_weights, expanded_values)

        # Combine heads and project back to the original dimension
        # [B, H, S, K] -> [B, S, H, K] -> [B, S, D]
        context_vect = (
            context_vect.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        with record_function("out_projection"):
            return self.out_proj(context_vect)


class TinyFeedForwardNetwork(nn.Module):
    def __init__(self, *, config: ModelConfig) -> None:
        super().__init__()

        # Expanding the hidden dimension gives the model more room to
        # learn richer intermediate representations.
        hidden_dim = 4 * config.d_model
        self.fc1 = nn.Linear(config.d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.d_model)
        self.act = nn.GELU(approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,  # Shape: [B, S, D]
    ) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyDecoderBlock(nn.Module):
    def __init__(self, *, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()

        self.attn = TinyMultiHeadAttention(config=config, layer_idx=layer_idx)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ffn = TinyFeedForwardNetwork(config=config)
        self.ln_2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,  # Shape: [B, S, D]
        *,
        cache: KVCacheProtocol | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache=cache)
        x = x + self.ffn(self.ln_2(x))

        return x


class TinyTransformer(nn.Module):
    def __init__(self, *, config: ModelConfig) -> None:
        super().__init__()

        self.input_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [
                TinyDecoderBlock(config=config, layer_idx=idx)
                for idx in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.input_embed.weight

    def forward(
        self,
        x: torch.Tensor,  # Shape: [B, S]
        *,
        cache: KVCacheProtocol | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        start_idx = cache.length if cache is not None else 0

        if start_idx + seq_len > self.pos_embed.num_embeddings:
            raise ValueError(
                "Input sequence length exceeds model's maximum sequence length"
            )

        positions = torch.arange(start_idx, start_idx + seq_len, device=x.device)
        x = self.input_embed(x) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x, cache=cache)

        x = self.ln_f(x)

        if cache is not None:
            cache.advance(num_new_tokens=seq_len)

        return self.lm_head(x)


def load_gpt2_weights(
    model: TinyTransformer, *, gpt2_state_dict: dict[str, torch.Tensor]
) -> None:
    first_attn = cast(TinyDecoderBlock, model.layers[0]).attn
    if first_attn.num_kv_heads != first_attn.num_query_heads:
        raise ValueError(
            "GPT-2 checkpoints use full multi-head attention; they cannot be "
            "loaded into a GQA/MQA model (num_kv_heads < num_query_heads) "
            "without converting or pooling the K/V projection weights."
        )

    max_seq_len = model.pos_embed.num_embeddings

    with torch.no_grad():
        model.input_embed.weight.copy_(gpt2_state_dict["wte.weight"])
        model.pos_embed.weight.copy_(gpt2_state_dict["wpe.weight"][:max_seq_len])

        d_model = model.input_embed.embedding_dim
        for i, module in enumerate(model.layers):
            layer = cast(TinyDecoderBlock, module)
            prefix = f"h.{i}."

            layer.ln_1.weight.copy_(gpt2_state_dict[prefix + "ln_1.weight"])
            layer.ln_1.bias.copy_(gpt2_state_dict[prefix + "ln_1.bias"])
            layer.ln_2.weight.copy_(gpt2_state_dict[prefix + "ln_2.weight"])
            layer.ln_2.bias.copy_(gpt2_state_dict[prefix + "ln_2.bias"])

            c_attn_weight = gpt2_state_dict[prefix + "attn.c_attn.weight"].T
            c_attn_bias = gpt2_state_dict[prefix + "attn.c_attn.bias"]
            q_w, k_w, v_w = c_attn_weight.split(d_model, dim=0)
            q_b, k_b, v_b = c_attn_bias.split(d_model, dim=0)

            layer.attn.Q_proj.weight.copy_(q_w)
            layer.attn.Q_proj.bias.copy_(q_b)
            layer.attn.K_proj.weight.copy_(k_w)
            layer.attn.K_proj.bias.copy_(k_b)
            layer.attn.V_proj.weight.copy_(v_w)
            layer.attn.V_proj.bias.copy_(v_b)

            layer.attn.out_proj.weight.copy_(
                gpt2_state_dict[prefix + "attn.c_proj.weight"].T
            )
            layer.attn.out_proj.bias.copy_(gpt2_state_dict[prefix + "attn.c_proj.bias"])

            layer.ffn.fc1.weight.copy_(gpt2_state_dict[prefix + "mlp.c_fc.weight"].T)
            layer.ffn.fc1.bias.copy_(gpt2_state_dict[prefix + "mlp.c_fc.bias"])
            layer.ffn.fc2.weight.copy_(gpt2_state_dict[prefix + "mlp.c_proj.weight"].T)
            layer.ffn.fc2.bias.copy_(gpt2_state_dict[prefix + "mlp.c_proj.bias"])

        model.ln_f.weight.copy_(gpt2_state_dict["ln_f.weight"])
        model.ln_f.bias.copy_(gpt2_state_dict["ln_f.bias"])
