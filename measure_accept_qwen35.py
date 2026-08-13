"""在标准数据集上测量 Qwen3.5-4B 的 baseline 与 DFlash 接受长度、吞吐。

用法: python measure_accept_qwen35.py [输出 json]

只比两条线:target 自回归 baseline,和 DFlash 的原生配置(block 16,不设 cap)。
关闭思考模式、greedy。MT-Bench 只用第一轮 —— 第二轮依赖第一轮的回答,单独作为
prompt 会失真。
"""

import json
import statistics
import sys
import time

import mlx_dspark as M

TARGET = "mlx-community/Qwen3.5-4B-8bit"
DRAFTER = "z-lab/Qwen3.5-4B-DFlash"
MAX_NEW = 256
# MT-Bench 共 80 条、HumanEval 共 164 条,都用满;GSM8K 有 1319 条,取 200 条控制耗时。
COUNTS = {"MT-Bench (chat)": 80, "HumanEval (code)": 164, "GSM8K (math)": 200}
OUT = sys.argv[1] if len(sys.argv) > 1 else "accept_qwen35.json"


def load_datasets():
    def jsonl(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    return {
        "MT-Bench (chat)": [r["turns"][0] for r in jsonl("data/mt_bench.jsonl")],
        "HumanEval (code)": ["Complete the following Python function:\n\n" + r["prompt"]
                             for r in jsonl("data/humaneval.jsonl")],
        "GSM8K (math)": [r["question"] for r in jsonl("data/gsm8k_test.jsonl")],
    }


def run(kind, target, tok, drafter, prompts):
    rounds, ntok, secs, per_prompt = [], 0, 0.0, []
    for p in prompts:
        ids = M.encode_messages(tok, [{"role": "user", "content": p}], enable_thinking=False)
        t0 = time.perf_counter()
        if kind == "baseline":
            r = M.greedy_generate(target, tok, prompt_ids=ids, apply_chat_template=False,
                                  max_new_tokens=MAX_NEW)
        else:
            r = M.dflash_generate(target, tok, drafter, prompt_ids=ids,
                                  apply_chat_template=False, max_new_tokens=MAX_NEW)
        dt = time.perf_counter() - t0
        acc = list(getattr(r, "accept_lengths", None) or [1])
        rounds += acc
        ntok += r.num_tokens
        secs += dt
        per_prompt.append(statistics.fmean(acc))
    return rounds, ntok, secs, per_prompt


def summarize(rounds, ntok, secs, per_prompt):
    mx = max(rounds)
    # P(接受 >= j),即逐位置命中率的累积乘积
    cum = [sum(1 for a in rounds if a >= j) / len(rounds) for j in range(1, mx + 1)]
    return {
        "mean_accept": round(statistics.fmean(rounds), 3),
        "median_accept": statistics.median(rounds),
        "stdev_across_rounds": round(statistics.stdev(rounds), 3) if len(rounds) > 1 else 0.0,
        "p_stdev_across_prompts": round(statistics.stdev(per_prompt), 3) if len(per_prompt) > 1 else 0.0,
        "min_prompt_mean": round(min(per_prompt), 2),
        "max_prompt_mean": round(max(per_prompt), 2),
        "prompts": len(per_prompt),
        "rounds": len(rounds),
        "tokens": ntok,
        "tok_s": round(ntok / secs, 1),
        "cum_accept_by_position": [round(c, 4) for c in cum],
    }


def main():
    ds = {k: v[:COUNTS[k]] for k, v in load_datasets().items()}
    print(f"target={TARGET}\ndrafter={DRAFTER}")
    print(f"max_new_tokens={MAX_NEW}  greedy  no-thinking  DFlash 原生 block 16(不设 cap)")
    print("  " + "  ".join(f"{k}={len(v)}条" for k, v in ds.items()) + "\n")
    results = {}

    tgt, tok, drf, _ = M.load_dflash_pair(TARGET, drafter=DRAFTER)
    probe = tok.decode(M.encode_messages(
        tok, [{"role": "user", "content": "hi"}], enable_thinking=False))
    print(f"prompt 尾部: {probe[-60:]!r}\n")

    for kind, label in (("baseline", "baseline"), ("dflash", "dflash block16")):
        for name, prompts in ds.items():
            out = summarize(*run(kind, tgt, tok, drf, prompts))
            results[f"{label} · {name}"] = out
            print(f"  {label:<14} {name:<20} 接受 {out['mean_accept']:6.3f}  "
                  f"({out['rounds']:5d} 轮, {out['tok_s']:6.1f} tok/s)", flush=True)

    meta = {"target": TARGET, "drafter": DRAFTER, "counts": COUNTS,
            "max_new_tokens": MAX_NEW, "greedy": True, "thinking": False,
            "cap": None, "dflash_block_size": 16}
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1, ensure_ascii=False)
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
