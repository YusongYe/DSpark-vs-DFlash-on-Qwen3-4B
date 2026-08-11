# DSpark vs DFlash on Qwen3-4B (Apple M4 Pro)

Head-to-head benchmark of two speculative-decoding drafters — DeepSeek's **DSpark** and z-lab's
**DFlash** — against the same `Qwen3-4B-8bit` target on Apple Silicon, via
[mlx-dspark](https://github.com/ARahim3/mlx-dspark).

**TL;DR — DSpark wins: 1.80× (84.8 tok/s) at cap 4. DFlash at its native 16-token block is a net
loss: 0.87×, slower than plain decoding.** Measured on an M4 Pro / mlx 0.32.0.

完整分析见 **[RESULTS.md](RESULTS.md)**。

---



