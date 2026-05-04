"""
cli_benchmark.py — Offline latency benchmark for the aggregated DistilGPT-2 model.
Measures tokens/sec without requiring the web server.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.llm_trainer import AGG_DIR

BENCHMARK_PROMPTS = [
    "The employee records department manages",
    "Clinical data privacy regulations require",
    "The synthetic data pipeline ensures",
    "Data retention policies for HR records state",
    "Membership inference attacks are used to verify",
]

def benchmark():
    print("=" * 60)
    print("  Sentinel-DS — Aggregated Model Latency Benchmark")
    print("=" * 60)

    if not __import__("os").path.isdir(AGG_DIR):
        print(f"\n[ERROR] No model found at {AGG_DIR}")
        print("  Run: python main.py --mode train\n")
        return

    print(f"\n[Benchmark] Loading model from {AGG_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(AGG_DIR)
    model = AutoModelForCausalLM.from_pretrained(AGG_DIR)
    model.eval()
    print("[Benchmark] Model loaded.\n")

    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(BENCHMARK_PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        t1 = time.perf_counter()

        new_tokens = output.shape[1] - input_len
        elapsed = t1 - t0
        tps = new_tokens / elapsed if elapsed > 0 else 0

        total_tokens += new_tokens
        total_time += elapsed

        print(f"  [{i+1}/{len(BENCHMARK_PROMPTS)}] Prompt: \"{prompt[:40]}...\"")
        print(f"         Generated {new_tokens} tokens in {elapsed:.2f}s → {tps:.1f} tok/s")
        print()

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print("=" * 60)
    print(f"  Total tokens: {total_tokens} | Total time: {total_time:.2f}s")
    print(f"  Average throughput: {avg_tps:.1f} tokens/sec")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()