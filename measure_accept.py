"""在标准数据集上测量 DSpark / DFlash 的平均接受长度。

用法: python measure_accept.py [每个数据集的 prompt 数] [输出 json]

三个数据集 (MT-Bench / HumanEval / GSM8K) 对应 chat / code / math 三类内容。
DSpark 关闭 n-gram 混合草稿 (lookup_drafts=False),否则接受长度会混入 n-gram
命中,与无 lookup 的 DFlash 路径不可比。
"""

import json
import statistics
import sys
import time

import mlx_dspark as M

TARGET = "mlx-community/Qwen3-4B-8bit"
MAX_NEW = 128
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else "accept_by_dataset.json"


def load_datasets(n):
    def jsonl(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    mt = [r["turns"][0] for r in jsonl("data/mt_bench.jsonl")][:n]
    he = [
        "Complete the following Python function:\n\n" + r["prompt"]
        for r in jsonl("data/humaneval.jsonl")
    ][:n]
    gsm = [r["question"] for r in jsonl("data/gsm8k_test.jsonl")][:n]
    return {"MT-Bench (chat)": mt, "HumanEval (code)": he, "GSM8K (math)": gsm}


def run(kind, target, tok, drafter, prompts, cap):
    """跑一组 prompt,返回逐轮接受长度与吞吐。"""
    rounds, ntok, secs, per_prompt = [], 0, 0.0, []
    for p in prompts:
        ids = M.encode_messages(tok, [{"role": "user", "content": p}], enable_thinking=False)
        t0 = time.perf_counter()
        if kind == "dspark":
            r = M.speculative_generate(
                target, tok, drafter, prompt_ids=ids, apply_chat_template=False,
                max_new_tokens=MAX_NEW, max_draft_tokens=cap, lookup_drafts=False,
            )
        else:
            r = M.dflash_generate(
                target, tok, drafter, prompt_ids=ids, apply_chat_template=False,
                max_new_tokens=MAX_NEW, max_draft_tokens=cap,
            )
        dt = time.perf_counter() - t0
        rounds += list(r.accept_lengths)
        ntok += r.num_tokens
        secs += dt
        per_prompt.append(statistics.fmean(r.accept_lengths) if r.accept_lengths else 0.0)
    return rounds, ntok, secs, per_prompt


def summarize(rounds, ntok, secs, per_prompt):
    mx = max(rounds)
    # P(接受 >= j) = 第 j 个位置的累计命中率,即理论里的 prod_{i<=j} p_i
    cum = [sum(1 for a in rounds if a >= j) / len(rounds) for j in range(1, mx + 1)]
    return {
        "mean_accept": round(statistics.fmean(rounds), 3),
        "median_accept": statistics.median(rounds),
        "p_stdev_across_prompts": round(statistics.stdev(per_prompt), 3) if len(per_prompt) > 1 else 0.0,
        "min_prompt_mean": round(min(per_prompt), 2),
        "max_prompt_mean": round(max(per_prompt), 2),
        "rounds": len(rounds),
        "tokens": ntok,
        "tok_s": round(ntok / secs, 1),
        "cum_accept_by_position": [round(c, 4) for c in cum],
    }


def main():
    ds = load_datasets(N)
    print(f"target={TARGET}  每数据集 {N} 条 prompt  max_new_tokens={MAX_NEW}  greedy  no-thinking")
    results = {}

    print("\n加载 DSpark pair …")
    tgt, tok, drf, _ = M.load_pair(TARGET)
    for name, prompts in ds.items():
        out = summarize(*run("dspark", tgt, tok, drf, prompts, 4))
        results[f"dspark cap4 · {name}"] = out
        print(f"  {name:<20} 平均接受 {out['mean_accept']:.3f}  ({out['rounds']} 轮, {out['tok_s']} tok/s)")
    del tgt, drf

    print("\n加载 DFlash pair …")
    tgt, tok, drf, _ = M.load_dflash_pair(TARGET)
    for cap, label in ((4, "cap4"), (None, "full16")):
        for name, prompts in ds.items():
            out = summarize(*run("dflash", tgt, tok, drf, prompts, cap))
            results[f"dflash {label} · {name}"] = out
            print(f"  {label:<7} {name:<20} 平均接受 {out['mean_accept']:.3f}  ({out['rounds']} 轮, {out['tok_s']} tok/s)")

    meta = {"target": TARGET, "n_prompts_per_dataset": N, "max_new_tokens": MAX_NEW,
            "greedy": True, "thinking": False, "lookup_drafts": False}
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1, ensure_ascii=False)
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
