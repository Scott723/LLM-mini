# LLM-mini

## 1. 模型配置

| Model | Parameters | Layers | Hidden Size | FFN Size | Q Heads | KV Heads | Head Dim | Seq Len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 19M | 19,161,600 | 8 | 256 | 768 | 4 | 2 | 64 | 512 |
| 300M | 303,163,392 | 20 | 1024 | 3072 | 16 | 8 | 64 | 512 |
| 800M | 813,376,512 | 26 | 1536 | 4608 | 24 | 12 | 64 | 512 |

共同配置：RoPE、RMSNorm、SwiGLU、GQA、BF16、weight tying。

---

## 2. 运行性能实验

### 2.1 Eager vs SDPA

19M 模型，`batch_size=32`，`gradient_accumulation_steps=2`，每个 optimizer step 处理 32,768 tokens。

| Backend | Stable Throughput | Peak Allocated | Peak Reserved | Stable MFU |
|---|---:|---:|---:|---:|
| Eager | ~171.2K tok/s | 13.07 GiB | 17.65 GiB | ~13.2% |
| SDPA | ~208.3K tok/s | 11.63 GiB | 16.27 GiB | ~16.1% |

结论：在相同训练设置下，SDPA 提高吞吐并降低峰值显存，后续训练默认使用 SDPA。

### 2.2 300M：不同 Micro-Batch

固定 `batch_size × gradient_accumulation_steps = 64`，每个 optimizer step 处理 32,768 tokens，使用 SDPA，不使用 Activation Checkpointing。

| Batch Size | GA | Stable Throughput | Peak Allocated | Peak Reserved | Stable MFU |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | ~31.6K tok/s | 12.60 GiB | 13.78 GiB | ~37.2% |
| 16 | 4 | ~36.6K tok/s | 20.10 GiB | 22.12 GiB | ~43.2% |
| 32 | 2 | ~33.7K tok/s | 35.09 GiB | 39.49 GiB | ~39.7% |

当前 300M 配置采用 `batch_size=16`、`gradient_accumulation_steps=4`。

### 2.3 800M：不同 Micro-Batch

固定 `batch_size × gradient_accumulation_steps = 64`，每个 optimizer step 处理 32,768 tokens，使用 SDPA，不使用 Activation Checkpointing。

| Batch Size | GA | Stable Throughput | Peak Allocated | Peak Reserved | Stable MFU |
|---:|---:|---:|---:|---:|---:|
| 2 | 32 | ~7.32K tok/s | 17.16 GiB | 18.02 GiB | ~22.7% |
| 4 | 16 | ~13.18K tok/s | 20.21 GiB | 21.46 GiB | ~40.9% |
| 8 | 8 | ~15.98K tok/s | 26.22 GiB | 28.06 GiB | ~49.6% |
| 16 | 4 | ~15.66K tok/s | 38.64 GiB | 42.43 GiB | ~48.6% |

100-step 验证中，`batch_size=8`、`GA=8` 的结果为：

| Metric | Result |
|---|---:|
| Average Throughput | 15.81K tok/s |
| Stable Throughput | ~15.9K tok/s |
| Peak Allocated | 26.22 GiB |
| Peak Reserved | 28.06 GiB |
| Average MFU | 49.0% |
| Stable MFU | ~49.3% |

当前 800M 配置采用 `batch_size=8`、`gradient_accumulation_steps=8`。

（图：不同模型 / batch size 下的 Throughput、Peak Memory、MFU 对比）

---

## 3. 后续实验

| Experiment | Setting | Main Comparison |
|---|---|---|
| Model Scaling | 19M / 300M / 800M；固定 seq_len=512、tokens/update=32,768、SDPA；使用各模型当前最佳 micro-batch | Validation Loss / PPL、Throughput、Peak Memory、MFU；按训练 tokens 和 wall-clock 对齐 |
| Depth vs Width | 约 300M 参数规模：27×896、20×1024、12×1280 | 在相近参数量下比较深窄、均衡、浅宽结构的 Loss / PPL、Throughput、Memory、MFU |
| MHA vs GQA vs MQA | 约 300M 基础结构；MHA: KV=16，GQA: KV=8，MQA: KV=1；其余结构保持不变 | Loss / PPL、训练 Throughput、Peak Memory、Generation Prefill/Decode Speed、KV Cache Memory |

（图：三个后续实验的实验设计示意图）
