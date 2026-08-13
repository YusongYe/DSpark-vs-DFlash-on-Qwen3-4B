"""第 1 步:用 target 自己生成响应,作为 drafter 的训练数据。

必须用 target 的输出而不是原始数据集里的答案 —— drafter 学的是"模仿这个 target
接下来会说什么",拿别的模型的文风去训会直接掉接受长度。

数据偏对话:Mac 上的实测显示 code 的条件命中率已经 0.946,几乎没有 suffix decay
可修,而 chat 只有 0.776 且是唯一不划算的场景。

严禁使用 MT-Bench —— 它是唯一的 chat 评测集。
"""

import argparse
import json
import os
import random

# 训练数据来源。刻意偏对话;都不含 MT-Bench。
SOURCES = [
    # (HF 数据集, split, 取出 prompt 的函数, 采样权重)
    ("HuggingFaceH4/ultrachat_200k", "train_sft",
     lambda r: r["messages"][0]["content"], 0.45),
    ("nvidia/Nemotron-Post-Training-Dataset-v2", "train",
     lambda r: r["messages"][0]["content"] if r.get("messages") else None, 0.35),
    ("sahil2801/CodeAlpaca-20k", "train",
     lambda r: r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else ""), 0.20),
]


def collect_prompts(n, seed=0):
    from datasets import load_dataset

    rng = random.Random(seed)
    prompts = []
    for name, split, extract, weight in SOURCES:
        want = int(n * weight)
        print(f"[data] {name} 目标 {want} 条 …", flush=True)
        try:
            ds = load_dataset(name, split=split, streaming=True)
        except Exception as e:
            print(f"[data] 跳过 {name}: {type(e).__name__}: {e}")
            continue
        got = []
        for row in ds:
            try:
                p = extract(row)
            except Exception:
                continue
            # 太短的没有信息量,太长的挤占生成预算
            if p and 16 <= len(p) <= 4000:
                got.append(p.strip())
            if len(got) >= want:
                break
        print(f"[data] {name} 实得 {len(got)} 条")
        prompts += got
    rng.shuffle(prompts)
    # 去重:重复 prompt 会让 Markov head 过拟合到少数 bigram
    seen, uniq = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default="data/responses.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompts = collect_prompts(args.n, args.seed)
    print(f"[data] 去重后共 {len(prompts)} 条 prompt")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    # 关闭思考模式:评测也是关的,而且思考文本啰嗦、不可预测,会拉低接受长度
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
        for p in prompts
    ]

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=8192)
    # greedy:评测也是 greedy,训练分布要对齐
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=args.max_new))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "w") as f:
        for p, o in zip(prompts, outs):
            resp = o.outputs[0].text
            # 太短的响应切不出几个 block,采集时会被丢掉,这里先过滤省得占地方
            if len(resp.strip()) < 64:
                continue
            f.write(json.dumps({"prompt": p, "response": resp}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[data] 写入 {kept} 条到 {args.out}")


if __name__ == "__main__":
    main()
