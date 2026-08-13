# 在 GCP 上训练 Qwen3.5-4B 的 DSpark Markov head

冻结 `z-lab/Qwen3.5-4B-DFlash` 的主干,只训练一个 rank-256 的 Markov head
(`markov_w1` / `markov_w2`,共 2 × 248320 × 256 = 1.27 亿参数),用来补上并行草稿
里缺失的"前一个 token 是什么"这一维信息。
