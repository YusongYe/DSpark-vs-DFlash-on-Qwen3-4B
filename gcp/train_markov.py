"""第 3 步:训练 Markov head。主干和 lm_head 全部冻结,只有两个矩阵有梯度。

Markov head 就是一个 rank-256 的 bigram 偏置(推理时的形式见 mlx-dspark
model.py 的 VanillaMarkov):

    step_bias(prev) = markov_w2(markov_w1(prev))          # [vocab] 的加性修正
    logits[i]       = base_logits[i] + step_bias(prev_i)

w2 初始化为全零,所以训练开始时 Markov head 是恒等无操作 —— 保证不会一上手就比
冻结主干更差。

推理时 prev 是模型自己上一步的 argmax,训练时若全用真实 token 就有 exposure bias。
--scheduled-sampling 按概率混入模型自己的预测;默认关闭(先拿 teacher forcing 基线)。
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_lm_head_weight(model_id, vocab_size, hidden_size):
    """只从 safetensors 里取出 embedding 矩阵,不加载整个 4B 模型。

    Qwen3.5 是 tie_word_embeddings=True,所以 lm_head 的权重就是 embed_tokens。
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    for st in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    w = f.get_tensor(k)
                    if tuple(w.shape) == (vocab_size, hidden_size):
                        print(f"[lm_head] {k} from {os.path.basename(st)} {tuple(w.shape)}")
                        return w
    raise RuntimeError(f"在 {model_id} 里找不到 [{vocab_size}, {hidden_size}] 的 embedding")


class MarkovHead(nn.Module):
    def __init__(self, vocab_size, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.markov_w1.weight, std=0.02)
        nn.init.zeros_(self.markov_w2.weight)      # 起点是恒等无操作

    def forward(self, prev_ids):
        return self.markov_w2(self.markov_w1(prev_ids))


class Shards(torch.utils.data.Dataset):
    """把所有分片摊平成 (hidden, prev, label, 位置序号) 的独立样本。

    teacher forcing 下各位置互相独立,可以完全并行 —— 这是默认路径。
    """

    def __init__(self, feats):
        self.files = sorted(glob.glob(os.path.join(feats, "shard_*.npz")))
        if not self.files:
            raise RuntimeError(f"{feats} 里没有 shard_*.npz")
        self.hidden, self.prev, self.label, self.pos = [], [], [], []
        for fp in self.files:
            z = np.load(fp)
            h, p, l = z["hidden"], z["prev"], z["label"]     # [N, B-1, H], [N, B-1]
            n, k, hid = h.shape
            self.hidden.append(h.reshape(n * k, hid))
            self.prev.append(p.reshape(n * k))
            self.label.append(l.reshape(n * k))
            self.pos.append(np.tile(np.arange(k, dtype=np.int16), n))
        self.hidden = np.concatenate(self.hidden)
        self.prev = np.concatenate(self.prev)
        self.label = np.concatenate(self.label)
        self.pos = np.concatenate(self.pos)
        print(f"[data] {len(self.files)} 个分片,共 {len(self.hidden)} 个训练位置")

    def __len__(self):
        return len(self.hidden)

    def __getitem__(self, i):
        return (torch.from_numpy(self.hidden[i]), int(self.prev[i]),
                int(self.label[i]), int(self.pos[i]))


def evaluate(head, lm_w, loader, device, block_size, decay):
    """返回 (base top-1, with-markov top-1, 逐位置 with-markov top-1)。

    top-1 命中率就是理论里的条件命中率 p,接受长度上限约 1/(1-p) —— 所以这个指标
    直接预测最终的接受长度。
    """
    head.eval()
    nb = nw = n = 0
    per_pos = np.zeros((block_size - 1, 2))
    with torch.inference_mode():
        for h, prev, lab, pos in loader:
            h = h.to(device, torch.bfloat16, non_blocking=True)
            prev, lab = prev.to(device), lab.to(device)
            base = h @ lm_w.T
            with_m = base + head(prev).to(base.dtype)
            b_ok = (base.argmax(-1) == lab)
            w_ok = (with_m.argmax(-1) == lab)
            nb += b_ok.sum().item()
            nw += w_ok.sum().item()
            n += lab.numel()
            for j in range(block_size - 1):
                m = (pos == j)
                if m.any():
                    per_pos[j, 0] += w_ok.cpu()[m].sum().item()
                    per_pos[j, 1] += m.sum().item()
    head.train()
    pp = np.where(per_pos[:, 1] > 0, per_pos[:, 0] / np.maximum(per_pos[:, 1], 1), np.nan)
    return nb / n, nw / n, pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats/")
    ap.add_argument("--target", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256, help="按位置计;logits 很宽,别开太大")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--decay-gamma", type=float, default=7.0,
                    help="靠后位置的 loss 权重衰减 (DFlash 论文 block16 用 7)")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--out", default="markov_head.safetensors")
    args = ap.parse_args()

    device = "cuda"
    meta = json.load(open(os.path.join(args.feats, "meta.json")))
    V, H, B = meta["vocab_size"], meta["hidden_size"], meta["block_size"]
    print(f"[cfg] vocab={V} hidden={H} block={B} rank={args.rank}")

    lm_w = load_lm_head_weight(args.target, V, H).to(device, torch.bfloat16)
    lm_w.requires_grad_(False)

    ds = Shards(args.feats)
    n_val = max(1, int(len(ds) * args.val_frac))
    tr, va = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                        num_workers=4, pin_memory=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=args.batch, num_workers=2)

    head = MarkovHead(V, args.rank).to(device)
    n_train = sum(p.numel() for p in head.parameters())
    print(f"[model] 可训练参数 {n_train / 1e6:.1f}M (主干与 lm_head 冻结)")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.04)
    # 位置权重:越靠后的位置对接受长度的边际贡献越小
    w_pos = torch.tensor(
        [float(np.exp(-j / args.decay_gamma)) for j in range(B - 1)], device=device)

    b0, w0, _ = evaluate(head, lm_w, dl_va, device, B, args.decay_gamma)
    print(f"[eval] 训练前  base top-1={b0:.4f}  with-markov={w0:.4f}  "
          f"(应当相等,w2 初始为零)")

    step = 0
    for ep in range(args.epochs):
        for h, prev, lab, pos in dl_tr:
            h = h.to(device, torch.bfloat16, non_blocking=True)
            prev, lab, pos = prev.to(device), lab.to(device), pos.to(device)
            with torch.inference_mode():
                base = h @ lm_w.T                      # 冻结,不需要梯度
            logits = base.clone().float() + head(prev).float()
            loss = (F.cross_entropy(logits, lab, reduction="none")
                    * w_pos[pos.long()]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"  ep{ep} step{step}/{steps} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        b, w, pp = evaluate(head, lm_w, dl_va, device, B, args.decay_gamma)
        ceil_b = 1 / max(1e-6, 1 - b)
        ceil_w = 1 / max(1e-6, 1 - w)
        print(f"[eval] ep{ep}  base p={b:.4f} (天花板 {ceil_b:.2f})  "
              f"with-markov p={w:.4f} (天花板 {ceil_w:.2f})")
        print(f"       逐位置 with-markov top-1: "
              + " ".join(f"{x:.3f}" for x in pp))

    from safetensors.torch import save_file
    save_file({"markov_head.markov_w1.weight": head.markov_w1.weight.detach().cpu(),
               "markov_head.markov_w2.weight": head.markov_w2.weight.detach().cpu()},
              args.out)
    print(f"[out] 写入 {args.out}")


if __name__ == "__main__":
    main()
