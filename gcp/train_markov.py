"""第 3 步:训练 Markov head。主干和 lm_head 全部冻结,只有两个矩阵有梯度。

Markov head 就是一个 rank-256 的 bigram 偏置(推理时的形式见 mlx-dspark
model.py 的 VanillaMarkov):

    step_bias(prev) = markov_w2(markov_w1(prev))          # [vocab] 的加性修正
    logits[i]       = base_logits[i] + step_bias(prev_i)

w2 初始化为全零,所以训练开始时 Markov head 是恒等无操作 —— 保证不会一上手就比
冻结主干更差。

推理时 prev 是模型自己上一步的 argmax,训练时若全用真实 token 就有 exposure bias。
默认是纯 teacher forcing(先拿基线)。

特征按分片流式读取:一次只把一个分片读进内存(20000 个 anchor 约 1.5 GB),所以
数据总量不受机器内存限制 —— g2-standard-4 只有 15 GB 内存,而 20000 条数据的特征
有 24 GB,全量载入会 OOM。
"""

import argparse
import glob
import json
import math
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


def load_shard(fp):
    """读一个分片并摊平成独立位置。teacher forcing 下各位置互相独立。"""
    z = np.load(fp)
    h, p, l = z["hidden"], z["prev"], z["label"]      # [N, B-1, H], [N, B-1]
    n, k, hid = h.shape
    return (h.reshape(n * k, hid), p.reshape(n * k), l.reshape(n * k),
            np.tile(np.arange(k, dtype=np.int16), n))


class Batcher:
    """按分片流式出 batch。一次只有一个分片在内存里。

    验证集优先整片留出(最后一个分片);分片太少时退化为在片内按下标切,
    保证冒烟测试(只有一个分片)也能跑。
    """

    def __init__(self, feats, batch, seed=0, val_frac=0.02):
        self.files = sorted(glob.glob(os.path.join(feats, "shard_*.npz")))
        if not self.files:
            raise RuntimeError(f"{feats} 里没有 shard_*.npz")
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        if len(self.files) >= 3:
            self.train_files, self.val_files = self.files[:-1], self.files[-1:]
            self.within = None
            print(f"[data] {len(self.files)} 个分片,末片留作验证")
        else:
            self.train_files = self.val_files = self.files
            self.within = val_frac
            print(f"[data] 只有 {len(self.files)} 个分片,片内按 {val_frac:.0%} 切验证集")

    def _idx(self, n, which):
        if self.within is None:
            return np.arange(n)
        cut = int(n * (1 - self.within))
        return np.arange(cut) if which == "train" else np.arange(cut, n)

    def iter(self, which="train", shuffle=True):
        """训练时丢掉尾巴上不满一个 batch 的部分;验证时不丢 —— 验证集可能整个都比
        一个 batch 小(冒烟测试只有一个分片时就是这样)。"""
        files = self.train_files if which == "train" else self.val_files
        order = self.rng.permutation(len(files)) if shuffle else np.arange(len(files))
        for fi in order:
            h, p, l, pos = load_shard(files[fi])
            idx = self._idx(len(h), which)
            if shuffle:
                self.rng.shuffle(idx)
            last = len(idx) - self.batch + 1 if which == "train" else len(idx)
            for s in range(0, max(last, 0), self.batch):
                sel = idx[s:s + self.batch]
                yield h[sel], p[sel], l[sel], pos[sel]
            del h, p, l, pos


def to_gpu(batch, device):
    h, p, l, pos = batch
    return (torch.from_numpy(h).to(device, torch.bfloat16, non_blocking=True),
            torch.from_numpy(p.astype(np.int64)).to(device),
            torch.from_numpy(l.astype(np.int64)).to(device),
            torch.from_numpy(pos.astype(np.int64)).to(device))


@torch.inference_mode()
def evaluate(head, lm_w, batcher, device, block_size):
    """返回 (base top-1, with-markov top-1, 逐位置 with-markov top-1)。

    top-1 命中率就是理论里的条件命中率 p,接受长度上限约 1/(1-p) —— 所以不用跑
    完整生成就能判断训练有没有效果。
    """
    head.eval()
    nb = nw = n = 0
    per_pos = np.zeros((block_size - 1, 2))
    for raw in batcher.iter("val", shuffle=False):
        h, prev, lab, pos = to_gpu(raw, device)
        base = h @ lm_w.T
        b_ok = (base.argmax(-1) == lab)
        w_ok = ((base + head(prev).to(base.dtype)).argmax(-1) == lab)
        nb += b_ok.sum().item()
        nw += w_ok.sum().item()
        n += lab.numel()
        pc = pos.cpu().numpy()
        wc = w_ok.cpu().numpy()
        for j in range(block_size - 1):
            m = pc == j
            if m.any():
                per_pos[j, 0] += wc[m].sum()
                per_pos[j, 1] += m.sum()
    head.train()
    if n == 0:
        raise RuntimeError("验证集为空 —— 分片太小或 batch 太大")
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

    batcher = Batcher(args.feats, args.batch, val_frac=args.val_frac)
    head = MarkovHead(V, args.rank).to(device)
    print(f"[model] 可训练参数 {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M "
          f"(主干与 lm_head 冻结)")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    # 总步数只是估算(分片级切分导致不精确),所以用会自己 clamp 的 lambda,
    # 步数超出估算时不会像 OneCycleLR 那样直接抛异常
    est = max(1, args.epochs * meta["positions"] // args.batch)
    warm = max(1, int(0.04 * est))

    def lr_at(s):
        if s < warm:
            return s / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, est - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    w_pos = torch.tensor([math.exp(-j / args.decay_gamma) for j in range(B - 1)],
                         device=device)

    b0, w0, _ = evaluate(head, lm_w, batcher, device, B)
    print(f"[eval] 训练前  base top-1={b0:.4f}  with-markov={w0:.4f}  "
          f"({'一致,接线正常' if abs(b0 - w0) < 1e-9 else '不一致 —— w2 初始为零,应当相等,检查接线'})")

    step = 0
    for ep in range(args.epochs):
        for raw in batcher.iter("train", shuffle=True):
            h, prev, lab, pos = to_gpu(raw, device)
            with torch.inference_mode():
                base = h @ lm_w.T                      # 冻结,不需要梯度
            logits = base.clone().float() + head(prev).float()
            loss = (F.cross_entropy(logits, lab, reduction="none") * w_pos[pos]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"  ep{ep} step{step}/~{est} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        if step == 0:
            raise RuntimeError(f"一个 batch 都没跑到 —— 数据太少,把 --batch 降到 "
                               f"{args.batch // 4} 以下再试")

        b, w, pp = evaluate(head, lm_w, batcher, device, B)
        print(f"[eval] ep{ep}  base p={b:.4f} (天花板 {1 / max(1e-6, 1 - b):.2f})  "
              f"with-markov p={w:.4f} (天花板 {1 / max(1e-6, 1 - w):.2f})")
        print("       逐位置 with-markov top-1: " + " ".join(f"{x:.3f}" for x in pp))

        from safetensors.torch import save_file
        save_file({"markov_head.markov_w1.weight": head.markov_w1.weight.detach().cpu(),
                   "markov_head.markov_w2.weight": head.markov_w2.weight.detach().cpu()},
                  args.out)
    print(f"[out] 写入 {args.out}")


if __name__ == "__main__":
    main()
