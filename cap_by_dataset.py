"""最优 cap 是否依赖内容类型?

在 HumanEval 上 DSpark cap4 的接受长度已达 4.33 / 上限 5(87% 饱和),说明代码类
内容的瓶颈已从"猜得准不准"转移到"cap 给不给得够"。本脚本逐数据集扫 cap,检验
最优 cap 是否随内容变化,以及 thinking 模式对接受长度的影响。
"""

import json
import statistics
import sys
import time

import mlx_dspark as M

TARGET = "mlx-community/Qwen3-4B-8bit"
MAX_NEW = 128
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def datasets(n):
    return {
        "MT-Bench (chat)": [r["turns"][0] for r in jsonl("data/mt_bench.jsonl")][:n],
        "HumanEval (code)": ["Complete the following Python function:\n\n" + r["prompt"]
                             for r in jsonl("data/humaneval.jsonl")][:n],
        "GSM8K (math)": [r["question"] for r in jsonl("data/gsm8k_test.jsonl")][:n],
    }


def run(target, tok, drafter, prompts, cap, thinking=False):
    rounds, ntok, secs = [], 0, 0.0
    for p in prompts:
        ids = M.encode_messages(tok, [{"role": "user", "content": p}], enable_thinking=thinking)
        t0 = time.perf_counter()
        r = M.speculative_generate(
            target, tok, drafter, prompt_ids=ids, apply_chat_template=False,
            max_new_tokens=MAX_NEW, max_draft_tokens=cap, lookup_drafts=False,
        )
        secs += time.perf_counter() - t0
        rounds += list(r.accept_lengths)
        ntok += r.num_tokens
    sat = statistics.fmean(rounds) / (cap + 1)          # 相对 cap+1 上限的饱和度
    return {"mean_accept": round(statistics.fmean(rounds), 3),
            "ceiling": cap + 1,
            "saturation": round(sat, 3),
            "tok_s": round(ntok / secs, 1),
            "rounds": len(rounds)}


def main():
    ds = datasets(N)
    tgt, tok, drf, _ = M.load_pair(TARGET)
    out = {}

    print(f"DSpark cap 扫描 · 每数据集 {N} 条 · max_new_tokens={MAX_NEW} · no-thinking\n")
    print(f"{'数据集':<20} {'cap':>4} {'接受长度':>9} {'上限':>5} {'饱和度':>7} {'tok/s':>8}")
    for name, prompts in ds.items():
        for cap in (2, 4, 5, 7):
            r = run(tgt, tok, drf, prompts, cap)
            out[f"dspark cap{cap} · {name}"] = r
            print(f"{name:<20} {cap:>4} {r['mean_accept']:>9.3f} {r['ceiling']:>5} "
                  f"{r['saturation']:>6.0%} {r['tok_s']:>8.1f}")
        print()

    print("=== thinking 模式的影响 (cap 4) ===")
    print(f"{'数据集':<20} {'thinking':>9} {'接受长度':>9} {'tok/s':>8}")
    for name, prompts in ds.items():
        for th in (False, True):
            r = run(tgt, tok, drf, prompts[: max(1, N // 2)], 4, thinking=th)
            out[f"dspark cap4 · {name} · thinking={th}"] = r
            print(f"{name:<20} {str(th):>9} {r['mean_accept']:>9.3f} {r['tok_s']:>8.1f}")

    with open("cap_by_dataset.json", "w") as f:
        json.dump({"meta": {"target": TARGET, "n": N, "max_new_tokens": MAX_NEW,
                            "lookup_drafts": False}, "results": out}, f, indent=1, ensure_ascii=False)
    print("\n写入 cap_by_dataset.json")


if __name__ == "__main__":
    main()
