# DSpark vs DFlash on Qwen3-4B (Apple M4 Pro)

在同一个 Qwen3-4B 目标模型上,对比 DeepSeek 的 **DSpark** 与 z-lab 的 **DFlash** 两个推测解码(speculative decoding)drafter,以及不做推测解码的 baseline。

---

## Summary

1. **DSpark 全面胜出。** 在 chat / code / math 三类 prompt 上都快于 DFlash,最好成绩 **1.80×**(84.8 tok/s,cap 4)。
2. **DFlash 在它的原生配置下是负收益。** 完整 16-token block 只有 **0.87×**,比不做推测解码还慢。它的最优点也在 cap 4(1.36×),但仍低于 DSpark。
3. **cap 是最重要的调参项,最优值是 4,不是默认的 2,也不是** `auto`**。** 手动 `--max-draft 4` 比 `auto`(1.49×)明显更快。
4. **cap 4 → 5 之间有一个拐点。**

```bash
mlx-dspark generate \
  --model mlx-community/Qwen3-4B-8bit \
  --mode dspark --max-draft 4 \
  --prompt "Explain KV cache."
```

---



## Env


|                  |                      |
| ---------------- | -------------------- |
| 芯片 / 内存          | Apple M4 Pro · 24 GB |
| 系统               | macOS 26.5.2 (25F84) |
| mlx              | 0.32.0               |
| mlx-lm / mlx-vlm | 0.31.3 / 0.6.12      |
| mlx-dspark       | 0.7.0                |
| Python           | 3.12.12              |




### 参与的三个 checkpoint

对比解码方式


| 角色             | Checkpoint                           | 层数  | hidden_size | 磁盘     | 使用于             |
| -------------- | ------------------------------------ | --- | ----------- | ------ | --------------- |
| target         | `mlx-community/Qwen3-4B-8bit`        | 36  | 2560        | 4.0 GB | 全部三种模式          |
| DSpark drafter | `deepseek-ai/dspark_qwen3_4b_block7` | 5   | 2560        | 2.6 GB | `--mode dspark` |
| DFlash drafter | `z-lab/Qwen3-4B-DFlash-b16`          | 5   | 2560        | 1.0 GB | `--mode dflash` |


---



## 一、模式对比(cap 2 与 auto)

3 次 trial 取中位数,256 tokens,greedy,covering chat / code / math。


| 运行                  | tok/s    | 加速        | 接受长度 | chat | code | math |
| ------------------- | -------- | --------- | ---- | ---- | ---- | ---- |
| baseline(纯 greedy)  | 43.1     | 1.00×     | 1.00 | 43.2 | 43.2 | 42.9 |
| DSpark cap=2        | 69.2     | 1.61×     | 2.27 | 66.6 | 65.1 | 76.0 |
| **DSpark cap=auto** | **72.4** | **1.68×** | 2.62 | 68.4 | 65.3 | 83.6 |
| DFlash cap=2        | 54.7     | 1.27×     | 2.08 | 51.6 | 53.3 | 59.1 |
| DFlash cap=auto     | 60.7     | 1.41×     | 2.39 | 52.8 | 57.7 | 71.5 |


分领域接受长度(cap=auto):


| 运行              | chat | code | math |
| --------------- | ---- | ---- | ---- |
| DSpark cap=auto | 2.37 | 2.29 | 3.20 |
| DFlash cap=auto | 2.06 | 2.26 | 2.86 |


math 类 prompt 最好预测(结构性最强),chat 最难 —— 这个规律在两个 drafter 上都成立。

---



## 二、cap 实验

`--max-draft`(cap)是每轮提交给目标模型验证的草稿 token 数上限。

### DSpark(block size = 7)

3 次 trial,**该轮 baseline = 47.0 tok/s**。


| cap   | tok/s    | 加速        | 接受长度 | 备注                   |
| ----- | -------- | --------- | ---- | -------------------- |
| 1     | 53.2     | 1.13×     | 1.76 | 单个草稿 token 约 76% 被接受 |
| 2     | 67.5     | 1.44×     | 2.27 | drafter 默认值          |
| 3     | 77.1     | 1.64×     | 2.52 |                      |
| **4** | **84.8** | **1.80×** | 2.74 | **最优**               |
| 5     | 68.8     | 1.46×     | 2.90 |                      |
| 7     | 60.4     | 1.29×     | 3.08 | acceptance最高         |
| auto  | 70.1     | 1.49×     | 2.68 | 偏保守                  |




### DFlash(block size = 16)

3 次 trial,**该轮 baseline = 43.7 tok/s**。


| cap   | tok/s    | 加速        | 接受长度 | 备注           |
| ----- | -------- | --------- | ---- | ------------ |
| 1     | 44.6     | 1.02×     | 1.68 | 基本打平         |
| 2     | 54.8     | 1.25×     | 2.08 |              |
| **4** | **59.6** | **1.36×** | 2.39 | **最优**       |
| 8     | 44.4     | 1.02×     | 2.57 | 回落至打平        |
| 16    | 38.2     | **0.87×** | 2.64 | **负收益**      |
| auto  | 59.0     | 1.35×     | 2.39 | 与 cap 4 基本重合 |


原始数据:`sweep_dspark.json`、`sweep_dflash.json`

Finding:

**1: 接受长度随 cap 单调上升,吞吐是倒 U 曲线。** cap 越大,drafter 猜中的机会越多; 但验证矩阵越宽、每轮越贵。两者相抵,吞吐在中间见顶。

**2: 接受长度的收益严重递减：** DFlash 的 cap 从 4 加到 16,宽度翻了四倍,接受只从 2.39 涨到 2.64(+10%),吞吐却掉 36%。

---



## 个人理解

DSpark = DFlash + rank-256 Markov head

Markov head试图解决 DFlash 那个"后面的位置不知道前面猜了什么"的缺陷

通过控制硬件和 target,所以剩下的差距只能来自算法本身。而且差距随 cap 扩大(0.08 → 0.19 → 0.35),正是 Markov head 解决的位置依赖问题。

---



## 复现方式

```bash
# 一次性环境准备(系统 Python 3.9 太旧)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-dspark
.venv/bin/mlx-dspark doctor

# 模式对比(baseline 始终作为参照一起跑)
.venv/bin/mlx-dspark benchmark --model mlx-community/Qwen3-4B-8bit \
  --modes dspark,dflash --trials 3 --max-new-tokens 256 --json bench_qwen3_4b.json

# cap 扫描
.venv/bin/mlx-dspark benchmark --model mlx-community/Qwen3-4B-8bit \
  --modes dspark --caps 1,2,3,4,5,7,auto --trials 3 --max-new-tokens 256 --json sweep_dspark.json

.venv/bin/mlx-dspark benchmark --model mlx-community/Qwen3-4B-8bit \
  --modes dflash --caps 1,2,4,8,16,auto --trials 3 --max-new-tokens 256 --json sweep_dflash.json

.venv/bin/mlx-dspark benchmark --model mlx-community/Qwen3-4B-8bit \
  --modes dspark --caps 2,4,5 --trials 5 --max-new-tokens 256 --json confirm_cap.json
```

单次生成(注意:冷启动的单次运行数字不可比 —— 冷跑 baseline 只有 38.1 tok/s,预热后是 43–47):

```bash
.venv/bin/mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --mode baseline --prompt "Explain KV cache."
.venv/bin/mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --mode dspark   --prompt "Explain KV cache."
.venv/bin/mlx-dspark generate --model mlx-community/Qwen3-4B-8bit --mode dflash   --prompt "Explain KV cache."
```

