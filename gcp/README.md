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

| 机型 | GPU | 系统内存 | 够不够 |
| --- | --- | --- | --- |
| `g2-standard-4` | 1× L4 24GB | 16 GB | 够,最省 |
| `g2-standard-8` | 1× L4 24GB | 32 GB | 够,宽松些 |
| `a2-highgpu-1g` | 1× A100 40GB | 85 GB | 宽裕,但没必要 |

系统内存不是瓶颈:特征按分片流式读,一次只有 1.5 GB 在内存里。

用 Spot 实例。GPU 时间的大头是数据生成(约 1 小时)和特征采集(约 1 小时),
Markov head 本身的训练是分钟级。**整个实验的 GPU 花费在十几美元量级。**

## 机器准备

先确认硬件够不够。在实例的终端里(控制台点 SSH 就是):

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
lspci | grep -i nvidia          # 驱动没装时用这个确认硬件在不在
df -h / ; free -g | head -2
```

三个硬性条件:**显存 ≥ 16 GB**(需要约 14 GB)、**compute capability ≥ 8.0**
(脚本全程 bf16;T4 是 7.5、V100 是 7.0,都不支持 bf16,会直接报错)、
**磁盘可用 ≥ 40 GB**(模型缓存 11 GB + 特征十几到二十几 GB)。

### 磁盘不够就扩启动盘

比挂新盘简单,而且可以在机器运行中扩。先在机器外面(本地或 Cloud Shell):

```bash
gcloud compute disks resize <盘名> --size=150GB --zone=<zone>
```

再回到实例里让文件系统认到新空间:

```bash
sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1 && df -h /
```

### 纯净镜像要自己装驱动

Deep Learning 镜像预装了驱动;普通 Debian/Ubuntu 镜像没有,用 Google 的官方脚本:

```bash
curl -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
sudo python3 install_gpu_driver.py      # 会编译内核模块,可能重启
```

**先扩盘再装驱动** —— 编译本身要占几 GB。

## 环境

```bash
# CUDA 12.x + Python 3.12
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install "transformers>=5.7" accelerate safetensors datasets numpy huggingface_hub

# z-lab 的 DFlash 推理代码(MIT),我们只用它的模型定义,不改
git clone https://github.com/z-lab/dflash ~/dflash && uv pip install -e ~/dflash
python -c "from dflash.model import DFlashDraftModel; print('dflash OK')"
```

vLLM 只有第 1 步用得到,而且它对 torch 版本挑,装在单独的 venv 里免得污染主环境:

```bash
uv venv --python 3.12 .venv-vllm && source .venv-vllm/bin/activate
uv pip install vllm "transformers>=5.7" datasets
```

注意 z-lab **没有开源训练代码**(README 里写的是 "will open-source the training
recipe soon"),仓库里只有推理。这里的训练循环是自己写的。

## 先跑冒烟测试

第 3 步的接线可以纯 CPU 验证,不用等前两步、也不用 GPU(用假特征跑一遍数据流、
训练循环和评测):

```bash
python test_train_markov.py
```

然后四步各用最小规模走一遍再上正式规模,不然很可能在第 3 步才发现第 1 步的问题,
一小时的 GPU 时间白烧。全程十几分钟。

```bash
mkdir -p data && curl -sL -o data/mt_bench.jsonl \
  https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl

source .venv-vllm/bin/activate && python gen_responses.py --n 40 --max-new 128 --out data/smoke.jsonl
source .venv/bin/activate
python collect_features.py --responses data/smoke.jsonl --anchors-per-seq 4 --out feats_smoke/
python train_markov.py --feats feats_smoke/ --epochs 1 --out /tmp/smoke.safetensors
python eval_markov.py --markov /tmp/smoke.safetensors --n 4 --max-new 64
```

要检查的是 `train_markov.py` 开头那行 **"训练前 base top-1 和 with-markov 一致"**。
`w2` 初始为零,两者必须相等;不相等说明接线错了。40 条数据训出来的 head 没有意义,
这步只验证管线。

正式跑记得用 `tmux`,SSH 断开不会中断。

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

第 2 步的输出很占地方:每个 anchor 是 `(block-1) × hidden × 2` 字节,block 16、
hidden 2560 下约 77 KB,所以 `20000 × 16` 个 anchor 是 **24 GB**。磁盘紧张就减
`--anchors-per-seq`(降到 8 就砍一半),别减 `--n` —— prompt 多样性比同一条序列上
多采几个位置更值钱。

内存不受这个数字影响:第 3 步一次只读一个分片(默认 20000 个 anchor,约 1.5 GB),
所以 15 GB 内存的机器也能训完 24 GB 的特征。

## 评测口径

和 Mac 上那轮保持一致,好直接对比:MT-Bench 前 80 条(全量)、每条 256 tokens、
greedy、关闭思考模式、DFlash 原生 block 16 不设 cap。**要打败的数字是 3.58。**

## 已知的坑

**exposure bias。** 推理时 Markov 偏置的 `prev` 是模型自己上一步的 argmax,训练时
用的是真实 token(teacher forcing),分布不匹配。这是整个实验最主要的不确定性 ——
DeepSeek 没公开怎么训的。当前脚本只做 teacher forcing,因为这样各位置互相独立、
可以完全并行,训练是分钟级的。如果 `eval_markov.py` 的接受长度涨幅明显小于第 3 步
验证集上 top-1 的涨幅,那基本就是这个原因,再加 scheduled sampling(按概率把 `prev`
换成模型自己的预测)。

**mlx-dspark 收不了这种 checkpoint。** 它的 `load_dflash` 显式检查 `markov_rank`
并报错,张量名也做严格比对。所以评测在 GCP 上用 PyTorch 做(`eval_markov.py`)。
想回 Mac 上测的话,需要给 mlx-dspark 的 `dflash_model.py` 打补丁,把它自己
`model.py` 里已有的 `VanillaMarkov` 接到 DFlash 路径上 —— 改动不大,但建议等
训练确实有效果之后再做。
