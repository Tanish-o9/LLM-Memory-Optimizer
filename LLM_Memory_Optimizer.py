"""
LLM Memory Optimizer - benchmark script
Ek hi model ko FP16 / INT8 / INT4 mein load karke memory, speed aur quality compare karta hai.

Run:
    python benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct --modes fp16 int8 int4
Output: results.csv (isse dashboard app.py padhta hai)

Note: bitsandbytes ke liye NVIDIA GPU chahiye (Colab free T4 chalta hai).
"""
import argparse
import gc
import math
import statistics
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Fixed prompts: har mode mein bilkul same prompts use honge (fair comparison)
PROMPTS = [
    "Explain what quantization means in machine learning in simple words.",
    "Write a short Python function that checks whether a number is prime.",
    "Give three tips to prepare for a software engineering interview.",
]


def load_model(name: str, mode: str):
    """Selected precision mein model load karta hai."""
    if mode == "fp16":
        kwargs = dict(torch_dtype=torch.float16, device_map={"": 0})
    elif mode == "int8":
        kwargs = dict(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
    elif mode == "int4":
        kwargs = dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map={"": 0},
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


@torch.no_grad()
def perplexity(model, tok, n_chunks: int = 20, ctx: int = 512) -> float:
    """Quality metric: WikiText-2 par perplexity (kam = behtar)."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    losses = []
    for i in range(n_chunks):
        chunk = ids[:, i * ctx : (i + 1) * ctx].to(model.device)
        if chunk.size(1) < ctx:
            break
        losses.append(model(chunk, labels=chunk).loss.float().item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def timed_generate(model, tok, prompt: str, max_new_tokens: int):
    """Ek generation chalata hai aur (seconds, new_tokens, text) return karta hai."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # har run mein same length
        do_sample=False,                # greedy = reproducible
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return dt, new_tokens, text


def benchmark_mode(name: str, mode: str, runs: int, max_new_tokens: int) -> dict:
    tok = AutoTokenizer.from_pretrained(name)

    # 1) Load time (alag se naapo)
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = load_model(name, mode)
    model.eval()
    torch.cuda.synchronize()
    load_time = time.perf_counter() - t0

    weights_mb = model.get_memory_footprint() / 1024**2

    # 2) Warm-up (timing mein count nahi hota)
    timed_generate(model, tok, PROMPTS[0], 8)

    # 3) Peak memory: ab se inference ka peak track karenge
    torch.cuda.reset_peak_memory_stats()

    # 4) First-token latency
    first_token = []
    for _ in range(runs):
        dt, _, _ = timed_generate(model, tok, PROMPTS[0], 1)
        first_token.append(dt)

    # 5) End-to-end latency + throughput (har prompt, repeated runs)
    latencies, tps, sample_output = [], [], ""
    for prompt in PROMPTS:
        for r in range(runs):
            dt, n, text = timed_generate(model, tok, prompt, max_new_tokens)
            latencies.append(dt)
            tps.append(n / dt)
            if not sample_output:
                sample_output = text

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # 6) Quality
    ppl = perplexity(model, tok)

    row = {
        "mode": mode,
        "load_time_s": round(load_time, 2),
        "weights_mb": round(weights_mb, 1),
        "peak_gpu_mb": round(peak_mb, 1),
        "first_token_ms": round(statistics.mean(first_token) * 1000, 1),
        "latency_s_mean": round(statistics.mean(latencies), 3),
        "latency_s_std": round(statistics.pstdev(latencies), 3),
        "tokens_per_s": round(statistics.mean(tps), 2),
        "perplexity": round(ppl, 3),
        "sample_output": sample_output.strip().replace("\n", " ")[:200],
    }

    # GPU saaf karo taaki agla mode fair measure ho
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--modes", nargs="+", default=["fp16", "int8", "int4"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU nahi mila. Colab T4 ya NVIDIA GPU use karo."
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for mode in args.modes:
        print(f"\n=== {mode} ===")
        rows.append(benchmark_mode(args.model, mode, args.runs, args.max_new_tokens))
        print({k: v for k, v in rows[-1].items() if k != "sample_output"})

    df = pd.DataFrame(rows)
    df.insert(0, "model", args.model)
    df["gpu"] = torch.cuda.get_device_name(0)
    df["torch"] = torch.__version__

    # Baseline (pehla mode) ke against reduction aur speedup
    base = df.iloc[0]
    df["memory_reduction_pct"] = ((base.peak_gpu_mb - df.peak_gpu_mb) / base.peak_gpu_mb * 100).round(1)
    df["speedup_x"] = (base.latency_s_mean / df.latency_s_mean).round(2)
    df["ppl_change"] = (df.perplexity - base.perplexity).round(3)

    df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")
    print(df.drop(columns=["sample_output"]).to_string(index=False))


if __name__ == "__main__":
    main()
