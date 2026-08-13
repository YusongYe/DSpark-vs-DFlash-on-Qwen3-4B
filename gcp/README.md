# 在 GCP 上训练 Qwen3.5-4B 的 DSpark Markov head

冻结 `z-lab/Qwen3.5-4B-DFlash` 的主干,只训练一个 rank-256 的 Markov head
(`markov_w1` / `markov_w2`,共 2 × 248320 × 256 = 1.27 亿参数),用来补上并行草稿
里缺失的"前一个 token 是什么"这一维信息。

## 为什么是这个实验

Mac 上的实测(见上级目录 `RESULTS.md`)给出的方向很明确:

| 数据集 | DFlash 接受长度 | 条件命中率 p | 天花板 1/(1-p) | 相对 baseline |
| --- | --- | --- | --- | --- |
| HumanEval (code) | 10.80 | 0.946 | 18.6 | 2.75× |
| GSM8K (math) | 6.74 | 0.865 | 7.4 | 2.18× |
| MT-Bench (chat) | 3.58 | 0.776 | 4.5 | **1.06×** |

code 上 p 已经 0.946,几乎没有 suffix decay 可修。**chat 是唯一有空间、也是唯一
不划算的场景**(1.06× 等于白干)。所以训练数据要偏对话,评测也盯 chat。

目标:把 chat 的 p 从 0.776 提到 0.85 左右,天花板从 4.5 涨到 6.7,接受长度进而
把 1.06× 拉到 1.8× 上下 —— 从"没用"变成"有用"。

## 机器

显存需求约 14 GB(target 9.3 + drafter 1.3 + 训练 2 + 激活 1~2),所以:

| 机型 | GPU | 够不够 |
| --- | --- | --- |
| `g2-standard-8` | 1× L4 24GB | 够,最省 |
| `a2-highgpu-1g` | 1× A100 40GB | 宽裕,推荐 |
| `a3-highgpu-1g` | 1× H100 80GB | 没必要 |

用 Spot 实例。GPU 时间的大头是数据生成(约 1 小时)和特征采集(约 1 小时),
Markov head 本身的训练是分钟级。**整个实验的 GPU 花费在十几美元量级。**

## 环境

```bash
# CUDA 12.x + Python 3.12
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install "transformers>=5.7" accelerate safetensors datasets numpy
uv pip install vllm                      # 只在第 1 步生成数据时需要

# z-lab 的 DFlash 推理代码(MIT),我们只用它的模型定义,不改
git clone https://github.com/z-lab/dflash /opt/dflash
export PYTHONPATH=/opt/dflash:$PYTHONPATH
```

注意 z-lab **没有开源训练代码**(README 里写的是 "will open-source the training
recipe soon"),仓库里只有推理。这里的训练循环是自己写的。

## 运行顺序

```bash
# 1) 用 target 自己生成响应。必须用 target 的输出做对齐,不能用原始数据集的答案。
#    偏对话数据;严禁使用 MT-Bench —— 它是唯一的 chat 评测集,污染了实验就废了。
python gen_responses.py --n 20000 --out data/responses.jsonl

# 2) 强制解码采集特征:每个 block 位置的 drafter 最终隐状态 + prev token + label
python collect_features.py --responses data/responses.jsonl \
    --anchors-per-seq 16 --out feats/

# 3) 训练 Markov head(只有两个矩阵有梯度)
python train_markov.py --feats feats/ --epochs 4 --out markov_head.safetensors

# 4) 评测:同一个主干,加/不加 Markov head 的接受长度对比
python eval_markov.py --markov markov_head.safetensors --n 80
```

## 评测口径

和 Mac 上那轮保持一致,好直接对比:MT-Bench 前 80 条(全量)、每条 256 tokens、
greedy、关闭思考模式、DFlash 原生 block 16 不设 cap。**要打败的数字是 3.58。**

## 已知的坑

**exposure bias。** 推理时 Markov 偏置的 `prev` 是模型自己上一步的 argmax,训练时
若全用真实 token(teacher forcing),分布不匹配。`train_markov.py` 里
`--scheduled-sampling` 可以按概率混入模型自己的预测。建议先跑纯 teacher forcing
拿基线,再对比。这是整个实验最主要的不确定性 —— DeepSeek 没公开怎么训的。

**mlx-dspark 收不了这种 checkpoint。** 它的 `load_dflash` 显式检查 `markov_rank`
并报错,张量名也做严格比对。所以评测在 GCP 上用 PyTorch 做(`eval_markov.py`)。
想回 Mac 上测的话,需要给 mlx-dspark 的 `dflash_model.py` 打补丁,把它自己
`model.py` 里已有的 `VanillaMarkov` 接到 DFlash 路径上 —— 改动不大,但建议等
训练确实有效果之后再做。
