"""Measure RadixArk/Qwen3.8-27B-DSpark against a running SGLang server.

Default sampling matches the model card so numbers are comparable to mean 3.39:
thinking on, temperature 0.6, top-k 20, top-p 0.95.

  python measure_qwen38_dspark.py                  # 每集 16 条, 512 tokens, 先看通不通
  python measure_qwen38_dspark.py --official       # 对齐 model card: 最多 128 条, 2048 tokens

Server must already be up, e.g.

  sglang serve --trust-remote-code \\
    --model-path Qwen/Qwen3.8-27B-FP8 \\
    --tp-size 1 --attention-backend fa3 \\
    --speculative-algorithm DSPARK \\
    --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \\
    --speculative-dspark-block-size 7 \\
    --speculative-draft-model-quantization unquant \\
    --mamba-scheduler-strategy extra_buffer \\
    --mem-fraction-static 0.8 --host 0.0.0.0 --port 30000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

DATASETS = {
    "MT-Bench (chat)": ("mt_bench.jsonl", 80),
    "HumanEval (code)": ("humaneval.jsonl", 164),
    "GSM8K (math)": ("gsm8k_test.jsonl", 200),
}


def load_prompts(data_dir: Path, name: str, n: int) -> list[str]:
    path = data_dir / DATASETS[name][0]
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if name.startswith("MT-Bench"):
        return [r["turns"][0] for r in rows][:n]
    if name.startswith("HumanEval"):
        return ["Complete the following Python function:\n\n" + r["prompt"] for r in rows][:n]
    return [r["question"] for r in rows][:n]


def wait_server(url: str, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/v1/models", timeout=3)
            if r.ok:
                print(f"server ready  {url}  {r.json()['data'][0]['id']}", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    sys.exit(f"SGLang not up at {url} after {timeout:.0f}s")


def generate(url: str, prompt_text: str, args) -> dict:
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new,
        },
    }
    r = requests.post(f"{url}/generate", json=payload, timeout=args.req_timeout)
    r.raise_for_status()
    return r.json()


def accept_of(meta: dict) -> float:
    if meta.get("spec_accept_length") is not None:
        return float(meta["spec_accept_length"])
    ntok = meta.get("completion_tokens") or 0
    nver = meta.get("spec_verify_ct") or 0
    if nver:
        return ntok / nver
    return 1.0


def run_dataset(url, tok, prompts, thinking, args, label):
    accepts, ntok, secs, missing = [], 0, 0.0, 0
    for i, p in enumerate(prompts, 1):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        t0 = time.perf_counter()
        body = generate(url, text, args)
        dt = time.perf_counter() - t0
        meta = body.get("meta_info") or {}
        acc = accept_of(meta)
        ctok = int(meta.get("completion_tokens") or 0)
        if meta.get("spec_accept_length") is None and not meta.get("spec_verify_ct"):
            missing += 1
        accepts.append(acc)
        ntok += ctok
        secs += dt
        if i % 4 == 0 or i == len(prompts):
            print(
                f"    {label} {i}/{len(prompts)}  accept {statistics.fmean(accepts):.2f}  "
                f"{ntok / max(secs, 1e-9):.1f} tok/s",
                flush=True,
            )
    return {
        "mean_accept": round(statistics.fmean(accepts), 3),
        "median_accept": round(statistics.median(accepts), 3),
        "min_accept": round(min(accepts), 3),
        "max_accept": round(max(accepts), 3),
        "prompts": len(prompts),
        "tokens": ntok,
        "tok_s": round(ntok / max(secs, 1e-9), 1),
        "seconds": round(secs, 1),
        "missing_spec_stats": missing,
        "card_ref": {
            "MT-Bench (chat)": 3.10,
            "HumanEval (code)": 3.47,
            "GSM8K (math)": 4.57,
        }.get(label),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--n", type=int, default=16, help="prompts per dataset (ignored with --official)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--greedy", action="store_true", help="temp=0; not comparable to the 3.39 card")
    ap.add_argument("--official", action="store_true", help="card protocol: ≤128 prompts, 2048 tokens")
    ap.add_argument("--wait", type=float, default=600)
    ap.add_argument("--req-timeout", type=float, default=600)
    ap.add_argument("--out", type=Path, default=Path("accept_qwen38_dspark.json"))
    args = ap.parse_args()

    if args.greedy:
        args.temperature = 0.0
        args.top_k = 1
        args.top_p = 1.0
    thinking = not args.no_thinking
    if args.official:
        args.max_new = 2048

    wait_server(args.url, args.wait)
    print(f"loading tokenizer {args.tokenizer} …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    results = {}
    print(
        f"thinking={thinking}  temp={args.temperature}  max_new={args.max_new}  "
        f"n={'official≤128' if args.official else args.n}",
        flush=True,
    )
    for name, (_, full) in DATASETS.items():
        n = min(128, full) if args.official else min(args.n, full)
        prompts = load_prompts(args.data_dir, name, n)
        print(f"\n{name}  n={len(prompts)}", flush=True)
        results[name] = run_dataset(args.url, tok, prompts, thinking, args, name)
        s = results[name]
        ref = f"  card {s['card_ref']:.2f}" if s["card_ref"] is not None else ""
        print(f"  mean accept {s['mean_accept']:.3f}{ref}  {s['tok_s']} tok/s", flush=True)

    blob = {
        "target": "Qwen/Qwen3.8-27B-FP8",
        "drafter": "RadixArk/Qwen3.8-27B-DSpark",
        "thinking": thinking,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new,
        "official_protocol": args.official,
        "results": results,
    }
    args.out.write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {args.out}")
    if any(s["missing_spec_stats"] for s in results.values()):
        print("WARNING: some responses had no spec_accept_length — server is probably AR-only")


if __name__ == "__main__":
    main()
