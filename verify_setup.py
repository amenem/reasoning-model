"""Check the chapter 2 setup end to end: device, tokenizer, model, generation.

Mirrors "Build a Reasoning Model (From Scratch)", sections 2.3 through 2.7.
Run download_model.py first.

    python verify_setup.py            # auto-selects CPU/GPU via get_device()
    python verify_setup.py --cpu      # force CPU (book's debugging override)
"""

import argparse
import time
from pathlib import Path

import torch
from reasoning_from_scratch.ch02 import generate_text_basic_stream, get_device
from reasoning_from_scratch.qwen3 import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
)

QWEN3_DIR = Path("qwen3")
PROMPT = "Explain large language models in a single sentence."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    print(f"PyTorch version {torch.__version__}")
    device = torch.device("cpu") if args.cpu else get_device()

    # Section 2.4: tokenizer round trip.
    tokenizer_path = QWEN3_DIR / "tokenizer-base.json"
    model_path = QWEN3_DIR / "qwen3-0.6B-base.pth"
    for path in (tokenizer_path, model_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python download_model.py")

    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    token_ids = tokenizer.encode(args.prompt)
    assert tokenizer.decode(token_ids) == args.prompt, "round trip failed"
    print(f"Tokenizer round trip OK ({len(token_ids)} tokens)")

    # Section 2.6: load the pretrained weights.
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Qwen3 0.6B base: {n_params:,} parameters on {device}")

    # Section 2.7 / listing 2.2: stream tokens until EOS.
    input_token_ids_tensor = torch.tensor(token_ids, device=device).unsqueeze(0)
    print(f"\nPrompt: {args.prompt}\nOutput:", end="", flush=True)

    start = time.time()
    n_generated = 0
    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        n_generated += 1
        print(tokenizer.decode(token.squeeze(0).tolist()), end="", flush=True)

    elapsed = time.time() - start
    rate = n_generated / elapsed if elapsed else float("nan")
    print(f"\n\n{n_generated} tokens in {elapsed:.1f}s ({rate:.1f} tok/s)")


if __name__ == "__main__":
    main()
