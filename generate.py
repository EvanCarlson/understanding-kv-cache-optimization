"""Load GPT-2 weights in FP32 and print a cached continuation of the input prompt."""

import argparse
import logging

import tiktoken
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model import (
    KVCacheFactory,
    KVCacheProtocol,
    ModelConfig,
    TinyTransformer,
    load_gpt2_weights,
)
from shared import DEFAULT_SEED, get_device, seed_everything

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "The quick brown fox jumps over"
DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_MODEL_REPO = "gpt2"
DEFAULT_CACHE_TYPE = "basic"
CACHE_TYPE_CHOICES = ["basic", "gqa", "quantized", "paged"]
ATTENTION_BACKEND_CHOICES = ["expanded", "grouped"]


def validate_optimization_options(*, cache_type: str, attention_backend: str) -> None:
    if attention_backend not in ATTENTION_BACKEND_CHOICES:
        raise ValueError(f"Unknown attention backend: {attention_backend}")
    if cache_type not in ("basic", "gqa", "none") and attention_backend != "expanded":
        raise ValueError(
            "Non-default --attention-backend settings require basic, gqa, or none"
        )


GPT2_SMALL_D_MODEL = 768
GPT2_SMALL_N_LAYERS = 12
GPT2_SMALL_NUM_HEADS = 12
GPT2_SMALL_MAX_POSITIONS = 1024


def load_pretrained_model(
    *, config: ModelConfig, model_repo: str, device: torch.device
) -> TinyTransformer:
    model = TinyTransformer(config=config).to(device=device, dtype=torch.float32)
    model.eval()

    logger.info("Downloading %s weights...", model_repo)
    weights_path = hf_hub_download(repo_id=model_repo, filename="model.safetensors")
    gpt2_state_dict = load_file(weights_path, device=str(device))
    load_gpt2_weights(model, gpt2_state_dict=gpt2_state_dict)

    return model


def generate(
    *,
    model: TinyTransformer,
    cache: KVCacheProtocol,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    logits = model(input_ids, cache=cache)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        logits = model(next_token, cache=cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    return torch.cat(generated, dim=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to continue")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repo to load GPT-2-shaped weights from",
    )
    parser.add_argument(
        "--cache-type",
        default=DEFAULT_CACHE_TYPE,
        choices=CACHE_TYPE_CHOICES,
        help="Which KVCacheProtocol implementation to decode with",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help=(
            "KV heads: a positive divisor of 12 (1 selects MQA). Defaults to "
            "3 for gqa, 12 otherwise. Reduced heads require --cache-type gqa "
            "and mean-pool GPT-2 K/V weights; output quality may change."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for reproducible generation",
    )
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="expanded",
        help="FP32 attention implementation (expanded or compact grouped K/V)",
    )
    args = parser.parse_args(argv)
    try:
        validate_optimization_options(
            cache_type=args.cache_type,
            attention_backend=args.attention_backend,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.num_kv_heads is None:
        args.num_kv_heads = 3 if args.cache_type == "gqa" else GPT2_SMALL_NUM_HEADS
    if args.num_kv_heads <= 0 or GPT2_SMALL_NUM_HEADS % args.num_kv_heads != 0:
        parser.error("--num-kv-heads must be a positive divisor of 12")
    if args.cache_type != "gqa" and args.num_kv_heads != GPT2_SMALL_NUM_HEADS:
        parser.error("reduced --num-kv-heads requires --cache-type gqa")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    seed_everything(seed=args.seed)
    device = torch.device(get_device())

    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    config = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        d_model=GPT2_SMALL_D_MODEL,
        n_layers=GPT2_SMALL_N_LAYERS,
        num_heads=GPT2_SMALL_NUM_HEADS,
        num_kv_heads=args.num_kv_heads,
        attention_backend=args.attention_backend,
        max_seq_len=input_ids.shape[1] + args.max_new_tokens,
    )
    model = load_pretrained_model(
        config=config, model_repo=args.model_repo, device=device
    )

    cache = KVCacheFactory.create(
        args.cache_type, config=config, batch_size=input_ids.shape[0], device=device
    )

    with torch.no_grad():
        generated_ids = generate(
            model=model,
            cache=cache,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
        )

    print("Prompt:", args.prompt)
    print("Continuation:", tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
