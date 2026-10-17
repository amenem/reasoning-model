# Build a Reasoning Model (From Scratch)

Working project for Sebastian Raschka's *Build a Reasoning Model (From Scratch)*.
Chapter 2 setup: the pretrained **Qwen3 0.6B base** model, loaded through the
book's pure-PyTorch reimplementation (`reasoning_from_scratch`) — no external LLM
libraries, no API calls, everything runs locally.

## Environment

The book offers a pip path and a `uv` path (section 2.2). This project uses pip
with a local virtual environment, since `uv` is not installed on this machine:

```bash
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install -r requirements.txt
```

`--system-site-packages` lets the venv reuse the torch already present in the
Anaconda base environment instead of re-downloading it.

Installed: torch 2.10.0, tokenizers 0.23.2, reasoning-from-scratch 0.2.0 on
Python 3.10.11. Device: Apple Silicon GPU (MPS) is available, so `get_device()`
selects `mps`; pass `--cpu` to the verify script to force CPU as the book
suggests while debugging.

## Download the model

```bash
./.venv/bin/python download_model.py
```

This calls the book's `download_qwen3_small(kind="base", tokenizer_only=False,
out_dir="qwen3")` and writes two files into `qwen3/`:

| File | Size | Source |
| --- | --- | --- |
| `tokenizer-base.json` | ~6 MB | Qwen3 0.6B base tokenizer |
| `qwen3-0.6B-base.pth` | ~1433 MB | Qwen3 0.6B base weights |

Options:

```bash
./.venv/bin/python download_model.py --tokenizer-only   # just the 6 MB tokenizer
./.venv/bin/python download_model.py --kind reasoning   # official reasoning variant
```

The `reasoning` variant is Qwen3's own chain-of-thought model, used later in the
book only as an evaluation reference — the base model is what the reasoning
techniques get built on.

## Verify the setup

```bash
./.venv/bin/python verify_setup.py
```

Runs the chapter 2 pipeline end to end: picks a device (listing 2.1), does a
tokenizer encode/decode round trip (section 2.4), loads the weights into
`Qwen3Model` with `QWEN_CONFIG_06_B` (section 2.6), and streams a response with
`generate_text_basic_stream` (listing 2.2), stopping at `<|endoftext|>`.

Verified output on this machine:

```
PyTorch version 2.10.0
Using Apple Silicon GPU (MPS)
Tokenizer round trip OK (10 tokens)
Loaded Qwen3 0.6B base: 751,632,384 parameters on mps

Prompt: Explain large language models in a single sentence.
Output: Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to perform
tasks such as answering questions, writing text, and even creating music.

36 tokens in 13.2s (2.7 tok/s)
```

That matches the book's section 2.7 sample output closely; small wording
differences across devices are expected from floating-point behaviour. The
751.6M parameter count is the 596M transformer plus the untied 151,936 x 1,024
output head.

Repeat runs reach ~6 tok/s once the model file is in the OS page cache; the
13.2s above was a cold first run. Either way this is slow by design at this
point in the book — there is no KV cache
yet, so every step re-processes the whole sequence. Chapter 2's later sections
add the cache (`generate_text_basic_stream_cache`) and `torch.compile`.

## Generate text in a notebook

```bash
./.venv/bin/jupyter lab generate_text.ipynb
```

`generate_text.ipynb` loads the model once, then exposes a `generate(prompt)`
helper so you can keep trying prompts without reloading the weights. It also
compares the two code paths from chapter 2 on the same prompt:

| | 40 tokens |
| --- | --- |
| `generate_text_basic_stream` (listing 2.2) | 8.30 s, 4 tok/s |
| `generate_text_basic_stream_cache` (listing 2.4) | 3.14 s, 12 tok/s |

The notebook is committed without cell outputs; run it to populate them.

## Layout

```
download_model.py    Fetch tokenizer + weights into qwen3/
verify_setup.py      Device, tokenizer, model load, and generation check
generate_text.ipynb  Interactive text generation, with/without KV cache
requirements.txt     torch, tokenizers, reasoning-from-scratch
qwen3/               Downloaded weights (git-ignored, ~1.4 GB)
```

`qwen3/` and `.venv/` are git-ignored, so the 1.4 GB of weights and the 585 MB
environment stay out of version control and are reproduced by the two commands
above.

## Reference

- Book repo: https://github.com/rasbt/reasoning-from-scratch
- Model architecture details: appendix C of the book
