"""第 4 步:评测。同一个冻结主干,加 / 不加 Markov head 的接受长度对比。

口径与 Mac 上那轮一致(见上级目录 RESULTS.md),好直接对比:
MT-Bench 全部 80 条、每条 256 tokens、greedy、关闭思考、DFlash 原生 block 16 不设 cap。
**要打败的数字是 3.58。**

生成循环照抄 z-lab 的 dflash_generate,唯一改动是草稿采样:原版对 block 内所有位置
并行取 argmax,加了 Markov head 之后要顺序走一遍,把上一位的预测喂给下一位。
这就是 DSpark 相对 DFlash 的全部区别。

先下载数据:
  curl -sL -o data/mt_bench.jsonl \
    https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
"""

import argparse
import json
import statistics

import torch


@torch.inference_mode()
def generate(target, drafter, embed, lm_head, input_ids, max_new_tokens,
             stop_token_ids, markov=None):
    """返回 (逐轮接受长度, 生成的 token 数)。markov=None 时退化为原版 DFlash。"""
    from transformers import DynamicCache

    from dflash.model import extract_context_feature

    B = drafter.block_size
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    dev = input_ids.device

    out_ids = torch.full((1, max_len + B), drafter.mask_token_id, dtype=torch.long, device=dev)
    position_ids = torch.arange(out_ids.shape[1], device=dev).unsqueeze(0)
    kv_t, kv_d = DynamicCache(), DynamicCache()

    o = target(input_ids, position_ids=position_ids[:, :n_in], past_key_values=kv_t,
               use_cache=True, logits_to_keep=1, output_hidden_states=True)
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in:n_in + 1] = o.logits.argmax(-1)
    target_hidden = extract_context_feature(o.hidden_states, drafter.target_layer_ids)

    accepts, start = [], n_in
    while start < max_len:
        block = out_ids[:, start:start + B].clone()
        h = drafter(
            target_hidden=target_hidden,
            noise_embedding=embed(block),
            position_ids=position_ids[:, kv_d.get_seq_length():start + B],
            past_key_values=kv_d, use_cache=True, is_causal=False,
        )[:, 1 - B:, :]
        draft_logits = lm_head(h)                      # [1, B-1, vocab]
        kv_d.crop(start)

        if markov is None:
            block[:, 1:] = draft_logits.argmax(-1)
        else:
            # 顺序链:每一位的偏置由上一位的预测决定 (推理时 prev 是自己的 argmax)
            prev = block[:, 0]
            for i in range(B - 1):
                nxt = (draft_logits[:, i, :] + markov(prev).to(draft_logits.dtype)).argmax(-1)
                block[:, i + 1] = nxt
                prev = nxt

        o = target(block, position_ids=position_ids[:, start:start + B],
                   past_key_values=kv_t, use_cache=True, output_hidden_states=True)
        posterior = o.logits.argmax(-1)
        n_acc = (block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()

        out_ids[:, start:start + n_acc + 1] = block[:, :n_acc + 1]
        out_ids[:, start + n_acc + 1] = posterior[:, n_acc]
        start += n_acc + 1
        kv_t.crop(start)
        accepts.append(n_acc + 1)
        target_hidden = extract_context_feature(
            o.hidden_states, drafter.target_layer_ids)[:, :n_acc + 1, :]

        if any(s in out_ids[:, n_in:start] for s in stop_token_ids):
            break

    return accepts, min(start + 1, max_len) - n_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--drafter", default="z-lab/Qwen3.5-4B-DFlash")
    ap.add_argument("--markov", default=None, help="markov_head.safetensors;不给则只测原版")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--mt-bench", default="data/mt_bench.jsonl")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from train_markov import MarkovHead

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map=dev, trust_remote_code=True).eval()
    drafter = AutoModel.from_pretrained(
        args.drafter, dtype=torch.bfloat16, device_map=dev, trust_remote_code=True).eval()
    embed, lm_head = target.get_input_embeddings(), target.get_output_embeddings()

    head = None
    if args.markov:
        from safetensors.torch import load_file

        w = load_file(args.markov)
        head = MarkovHead(drafter.config.vocab_size, args.rank).to(dev)
        head.markov_w1.weight.data = w["markov_head.markov_w1.weight"].to(dev, torch.float32)
        head.markov_w2.weight.data = w["markov_head.markov_w2.weight"].to(dev, torch.float32)
        head.eval()

    prompts = [json.loads(l)["turns"][0]
               for l in open(args.mt_bench) if l.strip()][:args.n]
    stop = [tok.eos_token_id]

    runs = {"DFlash (原版)": None}
    if head is not None:
        runs["DFlash + Markov head"] = head

    for label, m in runs.items():
        allacc, ntok = [], 0
        for p in prompts:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt").to(dev)
            acc, n = generate(target, drafter, embed, lm_head, ids,
                              args.max_new, stop, markov=m)
            allacc += acc
            ntok += n
        mean = statistics.fmean(allacc)
        print(f"{label:24s} 接受长度 {mean:.3f}  "
              f"(中位 {statistics.median(allacc)}, {len(allacc)} 轮, {ntok} tokens)")


if __name__ == "__main__":
    main()
