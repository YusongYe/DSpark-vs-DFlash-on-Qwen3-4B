"""CPU 冒烟测试:验证 Batcher / MarkovHead / evaluate 的接线,不碰真模型。"""
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from train_markov import Batcher, MarkovHead, evaluate, to_gpu

V, H, B, N, K = 500, 32, 5, 300, 4      # vocab, hidden, block, anchors/shard, B-1
d = "feats_test/"
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
rng = np.random.default_rng(0)
for s in range(4):
    np.savez(f"{d}/shard_{s:04d}.npz",
             hidden=rng.standard_normal((N, K, H)).astype(np.float16),
             prev=rng.integers(0, V, (N, K)).astype(np.int32),
             label=rng.integers(0, V, (N, K)).astype(np.int32))

bt = Batcher(d, batch=64)
assert len(bt.train_files) == 3 and len(bt.val_files) == 1, "分片级切分不对"
h, p, l, pos = next(iter(bt.iter("train")))
assert h.shape == (64, H) and set(np.unique(pos)) <= set(range(K)), (h.shape, np.unique(pos))
print("[ok] Batcher 形状与位置序号正常")

lm_w = torch.randn(V, H, dtype=torch.bfloat16)
head = MarkovHead(V, 8)
b0, w0, pp = evaluate(head, lm_w, bt, "cpu", B)
assert b0 == w0, f"w2 初始为零时两者必须相等: {b0} vs {w0}"
assert len(pp) == B - 1
print(f"[ok] 训练前 base==with-markov ({b0:.4f}), 逐位置 {len(pp)} 个")

opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
w_pos = torch.tensor([np.exp(-j / 7.0) for j in range(B - 1)], dtype=torch.float32)
n = 0
for raw in bt.iter("train"):
    h, prev, lab, pos = to_gpu(raw, "cpu")
    with torch.inference_mode():
        base = h @ lm_w.T
    loss = (F.cross_entropy(base.clone().float() + head(prev).float(), lab,
                            reduction="none") * w_pos[pos]).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    n += 1
assert head.markov_w2.weight.abs().sum().item() > 0, "w2 没有拿到梯度"
print(f"[ok] {n} 步训练跑通,w2 已更新,loss={loss.item():.3f}")

b1, w1, _ = evaluate(head, lm_w, bt, "cpu", B)
print(f"[ok] 训练后 base={b1:.4f} with-markov={w1:.4f} (随机标签,数值本身无意义)")

# 单分片路径:GCP 上的冒烟测试就是这种,验证集会比一个 batch 还小
d1 = "feats_test_one/"
shutil.rmtree(d1, ignore_errors=True)
os.makedirs(d1)
np.savez(f"{d1}/shard_0000.npz",
         hidden=rng.standard_normal((160, K, H)).astype(np.float16),
         prev=rng.integers(0, V, (160, K)).astype(np.int32),
         label=rng.integers(0, V, (160, K)).astype(np.int32))
b1 = Batcher(d1, batch=256)
n_val = sum(len(x[0]) for x in b1.iter("val", shuffle=False))
n_tr = sum(len(x[0]) for x in b1.iter("train"))
assert 0 < n_val < 256, f"验证集应当出一个不满的 batch: {n_val}"
assert n_tr > 0, "单分片时训练集为空"
evaluate(MarkovHead(V, 8), lm_w, b1, "cpu", B)
print(f"[ok] 单分片路径: 训练 {n_tr} 个位置, 验证 {n_val} 个 (小于 batch 也不丢)")

shutil.rmtree(d)
shutil.rmtree(d1)
