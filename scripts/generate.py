from __future__ import annotations

import argparse
from contextlib import nullcontext

import tiktoken
import torch

from myqwen.config import ModelConfig
from myqwen.generation import GenerationConfig, generate
from myqwen.modeling import CausalLM
from myqwen.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--eos-token-id", type=int, default=50256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = CausalLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt_ids = tokenizer.encode(args.prompt)

    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=args.eos_token_id,
        use_cache=not args.no_cache,
    )

    if args.precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 generation is enabled only on CUDA")
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        output_ids = generate(model, input_ids, generation_config)

    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
