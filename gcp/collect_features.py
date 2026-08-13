"""第 2 步:强制解码采集 Markov head 的训练特征。

因为主干是冻结的,采集就是"把真实 token 当 anchor 走一遍推理路径",不需要复现
DFlash 论文那套训练时的注意力掩码。张量构造严格照抄 z-lab `dflash_generate`:

  上下文特征 target_hidden = fused[0 : t]        (已提交的 token,位置 0..t-1)
  block      = [真实ids[t], mask, mask, ...]     (B 个,第 0 位是干净 anchor)
  position_ids = arange(0, t + B)                (K/V 是 ctx 和 block 拼接的)
  drafter 输出取后 B-1 位 → 这些就是要预测的位置

每个 anchor 产出 B-1 条样本,每条是 (drafter 隐状态, 前一个 token, 真实 token)。
不缓存 logits —— 248320 维一个位置就 0.5 MB;训练时用 target 的 lm_head 现算。
"""

import argparse
import json
import os

import numpy as np
import torch


def get_text_parts(target):
    """定位 embed_tokens 和 lm_head。

    Qwen3.5 是多模态封装 (Qwen3_5ForConditionalGeneration),层级比纯文本模型深一层,
    所以不能直接写死 target.model.embed_tokens。
    """
    lm_head = target.get_output_embeddings()
    embed = target.get_input_embeddings()
    if lm_head is None or embed is None:
        raise RuntimeError("找不到 lm_head / embed_tokens,检查 target 的封装层级")
    return embed, lm_head


def fused_context(target, input_ids, layer_ids):
    """跑一次 target 全序列前向,取指定层隐状态拼接成上下文特征。"""
    from dflash.model import extract_context_feature

    out = target(input_ids, output_hidden_states=True, use_cache=False)
    return extract_context_feature(out.hidden_states, layer_ids)


@torch.inference_mode()
def collect_one(drafter, embed, fused, ids, anchors, block_size, mask_token_id, device):
    """对一条序列的多个 anchor 采集特征。返回 (hidden, prev, label) 三个列表。"""
    from transformers import DynamicCache

    H, P, L = [], [], []
    for t in anchors:
        target_hidden = fused[:, :t, :]
        block = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        block[0, 0] = ids[0, t]                      # 第 0 位是干净 anchor
        noise_embedding = embed(block)
        position_ids = torch.arange(t + block_size, device=device).unsqueeze(0)

        h = drafter(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size:, :]                     # [1, B-1, hidden]

        H.append(h[0].to(torch.float16).cpu().numpy())
        # 位置 i 的草稿以 block 内第 i-1 个 token 为条件 (teacher forcing 用真实 token)
        P.append(ids[0, t:t + block_size - 1].cpu().numpy())
        L.append(ids[0, t + 1:t + block_size].cpu().numpy())
    return H, P, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--drafter", default="z-lab/Qwen3.5-4B-DFlash")
    ap.add_argument("--responses", default="data/responses.jsonl")
    ap.add_argument("--anchors-per-seq", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--shard-size", type=int, default=20000, help="每个分片多少个 anchor")
    ap.add_argument("--out", default="feats/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map=device, trust_remote_code=True).eval()
    drafter = AutoModel.from_pretrained(
        args.drafter, dtype=torch.bfloat16, device_map=device, trust_remote_code=True).eval()

    embed, _ = get_text_parts(target)
    block_size = drafter.block_size
    mask_token_id = drafter.mask_token_id
    layer_ids = drafter.target_layer_ids
    print(f"[cfg] block_size={block_size} mask_token_id={mask_token_id} "
          f"target_layer_ids={layer_ids}")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    buf_h, buf_p, buf_l = [], [], []
    shard, total = 0, 0

    def flush():
        nonlocal shard, buf_h, buf_p, buf_l
        if not buf_h:
            return
        np.savez(os.path.join(args.out, f"shard_{shard:04d}.npz"),
                 hidden=np.stack(buf_h), prev=np.stack(buf_p).astype(np.int32),
                 label=np.stack(buf_l).astype(np.int32))
        print(f"[out] shard_{shard:04d}.npz  {len(buf_h)} anchors", flush=True)
        shard += 1
        buf_h, buf_p, buf_l = [], [], []

    with open(args.responses) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"[in] {len(rows)} 条响应")

    for n, row in enumerate(rows):
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        resp_ids = tok(row["response"], return_tensors="pt",
                       add_special_tokens=False).input_ids
        ids = torch.cat([prompt_ids, resp_ids], dim=1)[:, :args.max_len].to(device)
        n_prompt, seq_len = prompt_ids.shape[1], ids.shape[1]

        # anchor 必须落在 response 区间内,且整个 block 要在序列范围内
        lo, hi = n_prompt, seq_len - block_size
        if hi <= lo:
            continue
        k = min(args.anchors_per_seq, hi - lo)
        anchors = rng.choice(np.arange(lo, hi), size=k, replace=False)

        fused = fused_context(target, ids, layer_ids)
        H, P, L = collect_one(drafter, embed, fused, ids, anchors,
                              block_size, mask_token_id, device)
        buf_h += H
        buf_p += P
        buf_l += L
        total += len(H)

        if len(buf_h) >= args.shard_size:
            flush()
        if n % 200 == 0:
            print(f"[{n}/{len(rows)}] 累计 {total} anchors "
                  f"({total * (block_size - 1)} 个训练位置)", flush=True)

    flush()
    meta = {"target": args.target, "drafter": args.drafter, "block_size": block_size,
            "mask_token_id": mask_token_id, "target_layer_ids": list(layer_ids),
            "anchors": total, "positions": total * (block_size - 1),
            "hidden_size": drafter.config.hidden_size,
            "vocab_size": drafter.config.vocab_size}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] {total} anchors → {total * (block_size - 1)} 个训练位置")


if __name__ == "__main__":
    main()
