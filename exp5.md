复现 Table 2 的核心目的在于通过消融实验（Ablation Study）定量验证不同数量的初始 token 作为 Attention Sink 对模型长文本困惑度（Perplexity, PPL）的影响 。

针对 Llama-2-7B，Table 2 的设定总 Cache 大小为 4096 。以下是为你设计的详细实验方案。

### 一、 实验目标与分组

评估 Llama-2-7B 模型在处理连续 65K tokens 的长文本时，不同 Attention Sink 数量对语言建模困惑度的影响 。

**核心实验组（共 5 组）：**
你需要保持总 KV Cache 容量恒定为 4096，测试以下 $x+y$（初始 token 数 + 最近 token 数）的配置 ：

1. 
**0+4096** (纯 Window Attention)：预期 PPL 发生剧烈崩坏（论文参考值 3359.95） 。


2. 
**1+4095**：预期 PPL 大幅下降但未完全恢复（论文参考值 11.88） 。


3. 
**2+4094**：预期 PPL 进一步改善（论文参考值 10.51） 。


4. 
**4+4092** (StreamingLLM 默认配置)：预期 PPL 恢复至健康状态（论文参考值 9.59） 。


5. 
**8+4088**：验证边际效应递减（论文参考值 9.54） 。



---

### 二、 环境与数据准备

* **模型:** `data/pretrained_models/Llama-2-7b-hf`。建议在 PyTorch 中强制指定 `device="mps"` 以利用硬件加速，并在长序列自回归循环中显式调用 `torch.mps.empty_cache()` 防止显存泄露。
* 
**数据集:** `/home/raosongde/streamingllm-experiment/data/pg19_books_65133.txt` 




---

### 三、 评估指标计算

实验的唯一量化指标是困惑度 (Perplexity) 。
在 65K tokens 的自回归过程中：

1. 记录每一步输出 logits 对目标真实 token 的 CrossEntropy Loss。
2. 累加这 400K 次的 Loss，求平均值（NLL, Negative Log-Likelihood）。
3. 最终计算公式：$PPL = \exp(\text{Average NLL})$。
