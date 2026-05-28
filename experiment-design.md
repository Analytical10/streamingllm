

## 🛠️ 第一阶段：环境与基础数据准备

### 1. 依赖环境安装

安装最新版本的 Hugging Face 库（确保其原生支持 `Cache` 架构与 `SinkCache`）及评测所需的基准库。

```bash
pip install transformers accelerate datasets lm-eval matplotlib juice-box

```

### 2. 实验语料下载与预处理

* 
**长文本语料（实验 1, 2）**：下载 **PG-19 测试集** 。


* 
**处理脚本**：编写一个 `prepare_pg19.py`，读取测试集中的前几本书，将其文本内容完全拼接（Concatenate），直到总长度用 Llama-2 的 Tokenizer 编码后超过 **20,000 个 Token** 。将其保存为 `pg19_20k.txt`。




* **流式任务语料（实验 3）**：
* 
**ARC 任务**：通过 Hugging Face `datasets` 下载 `ai2_arc` (Easy 和 Challenge) 。


* 
**StreamEval 干扰任务（核心自建）**：编写脚本，生成 100 个超长文本样本 。每隔 10 行插入一条随机键值对（例如：`The REGISTER_CONTENT in line 20 is <45603>`） ，并在 20 行之后紧跟对应的 Query 。总行数需保证文本长度达到 **120K Token** 。





---

## ⚙️ 第二阶段：核心实验脚本与测试套件编写

在撰写具体实验前，需要在本地建立一个统一的评测底座。核心任务是编写一个**流式 Token 预测与单步损失（Loss）收集函数**。

创建 `eval_utils.py`：

```python
import torch
import torch.nn as nn

def compute_streaming_loss(model, input_ids, kv_cache=None, custom_position_fn=None):
    """
    逐 Token 输入模型，模拟流式长文本解码，并收集每一步预测下一个 Token 的 CrossEntropy Loss。
    """
    model.eval()
    losses = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    # 转换为 batch_size=1 的 Tensor
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        for i in range(seq_len - 1):
            cur_token = input_ids[:, i:i+1]    # 当前输入的 token
            next_token = input_ids[:, i+1:i+2]  # 真实的目标下一个 token
            
            # 如果提供了动态修改 position_ids 的函数（实验2所需）
            position_ids = None
            if custom_position_fn is not None and kv_cache is not None:
                position_ids = custom_position_fn(kv_cache)
            
            # 前向传播
            outputs = model(
                input_ids=cur_token,
                past_key_values=kv_cache,
                position_ids=position_ids,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :] # 形状: [1, vocab_size]
            kv_cache = outputs.past_key_values # 滚动更新 cache
            
            # 计算当前步的预测损失
            step_loss = loss_fct(logits, next_token.view(-1))
            losses.append(step_loss.item())
            
    return losses

```

---

## 📈 第三阶段：四大核心实验详细执行步骤

### 实验 1：长文本下的 Perplexity (PPL) 基线测试与恢复

**目标**：复现图 3 。验证普通滑窗（Window Attention）在剔除首个 Token 后的瞬间崩溃 ，以及 StreamingLLM 引入 4 个 Sink Token 后的长效稳定 。

#### 实现步骤：

1. 
**加载模型**：初始化 `Llama-2-7b-hf`，使用 `float16` 载入 GPU 。


2. 
**加载数据**：读取 `pg19_20k.txt` 并将其转换为 Token IDs 。


3. **配置 4 组对比 Cache 进行测试**：
* **组 A (Dense Attention)**：不限制 Cache。直接调用 `compute_streaming_loss(..., kv_cache=None)`。
* 
**组 B (Window Attention)**：使用 HF 原生的 `DynamicCache`，但在每一步推理后，强制对其 Tensor 进行切片，只保留最近的 $L=2048$ 个 Token 。


* 
**组 C (StreamingLLM)**：直接使用 HF 的 `SinkCache(window_length=2048, num_sink_tokens=4)` 。


* 
**组 D (Re-computation)**：不传 Cache，每次预测都将当前 Token 及其前方最长 2048 跨度的所有历史 Token 重新打包装入模型（注意：此组运行极慢，可仅测前 5K 长度作为参考） 。




4. **计算累计 PPL 并绘图**：
* 对采集到的单步 `losses` 数组进行滑动窗口平均（例如每 100 步取均值），使用公式 $\text{PPL} = \exp(\text{mean}(\text{Loss}))$ 计算该阶段的 PPL 。


* 以 Token 输入长度（0 到 20K）为 X 轴，PPL 值为 Y 轴，绘制折线图，观察并验证组 B 是否在序列长度超过 2048 时出现 PPL 的突变飙升 。





---

### 实验 2：位置编码（Positional Embedding）的缓存对齐实验

**目标**：复现第 5 页关于位置编码的论述 。证明当 Cache 被截断后，RoPE 必须基于“**Cache 内部相对位置**”进行旋转，绝对位置会导致模型失效 。

#### 实现步骤：

1. **自定义位置计算函数**：在实验脚本中编写两个控制函数。
* **对齐函数（StreamingLLM 标准做法）**：
```python
def aligned_position_ids_fn(kv_cache):
    cache_len = kv_cache.get_seq_length(layer_idx=0)
    if cache_len < 2048:
        return torch.tensor([[cache_len]], dtype=torch.long, device="cuda")
    # 满窗口后，强制新 Token 永远使用窗口上界位置索引（L-1）
    return torch.tensor([[2047]], dtype=torch.long, device="cuda")

```


* **非对齐函数（对照组）**：
```python
def misaligned_position_ids_fn(kv_cache):
    global_step_tracker += 1 # 维护一个全局绝对步数
    # 尽管 Cache 被裁剪到了 2048，但仍给新 Token 分配其在全文本里的绝对序号
    return torch.tensor([[global_step_tracker]], dtype=torch.long, device="cuda")

```




2. **运行评测**：分别使用这两个函数配合 `StreamingLLMCache`（实验 1 中编写的裁剪类），在 `pg19_20k.txt` 数据集上跑通流式 Loss 收集。
3. 
**结果对比**：计算两个策略下的整体 PPL 。你会复现出：使用绝对位置的非对齐组 PPL 会大幅升高，无法恢复到 5.x 级别的正常水平 ，从而论证 Cache 内相对位置分配对 RoPE 机制的必要性 。



---

### 实验 3：下游流式长任务验证 (StreamEval 与 LongBench)

**目标**：复现图 9 和 Table 8 。测试在长依赖干扰场景下，StreamingLLM 的局部上下文提取和流式任务回答表现 。

#### 步骤 3a：复现 StreamEval 多轮流式 Q&A 实验

1. 
**加载模型**：使用指令微调模型 `Llama-2-7b-chat-hf` 。


2. 
**配置流式长输入**：将准备好的 120K Token 的 `StreamEval` 语料灌入模型 ，使用 `SinkCache(window_length=2048, num_sink_tokens=4)` 运行流式解码循环 。


3. 
**拦截并记录 Answer**：每当循环遇到 `Query:` 标记时，让模型停下流式灌入，进入 **Generation 模式**生成接下来的 Token 直到遇到 Stop Token 结束 。


4. 
**准确率统计**：将模型生成的字符串与标准答案（前 20 行对应的键值）进行**精确字符串匹配（Exact Match）** 。绘制随着输入 Token 长度增长（由 0K 到 120K），模型单步回答准确率的趋势，复现类似图 9 的水平直线（保持约 80%+ 准确率） 。



#### 步骤 3b：复现 LongBench 长文档测试与“首尾保留”修复实验

1. 
**基线运行（Truncation 1750+1750）**：读取 LongBench 数据集中的 NarrativeQA 样本 。按照官方标准，截取长文档的前 1750 个 Token 和后 1750 个 Token 拼接 ，送入模型生成答案，计算得分（F1 / 粗准确率）作为 Baseline 。


2. 
**对比组测试（StreamingLLM 4+3496）**：直接把全量长文档（可能上万 Token）通过 `SinkCache(window_length=3500, num_sink_tokens=4)` 流式喂给模型 ，最后发起提问。记录其得分，会发现结果大幅暴跌（因为文章开头的关键叙事被滑窗冲掉了） 。


3. 
**修复验证组（StreamingLLM 1750+1750）**：将 Cache 策略修改为固定保留前 **1750 个 Token 作为 Attention Sink**，滚动滑窗留给后 1750 个 Token 。再次运行任务，验证其得分会完美修复，恢复到甚至微幅超越物理截断基线的水平 。



---

### 实验 4：带“汇 Token（Sink Token）”的预训练实验

**目标**：从零预训练 160M 小模型，复现 Table 3 。证明在预训练时强制加入 1 个专用可学习的虚拟 Token，能够实现在流式测试时**仅需 1 个 Sink Token** 就能稳住 PPL 的效果 。

#### 实现步骤：

1. **数据改造与多版本模型预训练**：
基于 `Pythia-160M` 架构与 LLaMA-Factory 的预训练阶段（`stage: pt`），准备 3 个独立的增量训练分支 ：


* 
**分支一（Vanilla 模型）**：直接使用标准语料进行 Full 参数预训练 。


* 
**分支二（Zero Sink 模型）**：修改模型源码中的 attention 模块（如 `modeling_pythia.py`），找到注意力计算的 softmax 位置，将其替换为 $Softmax_1$ 函数（即在分母里手动加 1：`scores.exp() / (scores.exp().sum() + 1.0)`） 。


* 
**分支三（Learnable Sink 模型）**：修改数据加载脚本。对每一个读入的训练文本 Block，**在最开头硬编码拼接一个固定的特殊 Placeholder Token**（如 `<sink_token>`，该 Token 需要在词表中注册并允许其 Embedding 随机初始化并参与训练梯度更新） 。




2. 
**统一训练约束**：三个分支保持完全一致的 Batch Size、Learning Rate 调度与相同的 Step 数（论文中为 143k 步）进行训练，并记录监控其 Loss 收敛曲线（复现图 6，保证收敛趋势一致） 。


3. **流式能力验收测试**：
训练完成后，使用三个模型分别进行流式评测，计算在 PG-19 数据集上的 PPL 。在测试时，改变参数 `num_sink_tokens` ：


* 测试配置一：`0 + 1024`（普通滑窗） 


* 测试配置二：`1 + 1023`（只保留 1 个初始 Token） 


* 测试配置三：`4 + 1020`（保留 4 个初始 Token） 




4. **复现 Table 3 结论**：
运行并记录结果表格 。你会观测到：Vanilla 和 Zero Sink 在配置二（`1 + 1023`）下，由于缺乏足够大的注意力吞噬分母，PPL 会显著崩塌或升高 ；而 **Learnable Sink 分支训练出的模型，在 `1 + 1023` 的测试下，PPL 就能直接恢复到 18.x 的完美低位**，且与 `4 + 1020` 效果无异 ！这也最终论证了“通过增加一个专属的可学习 Sink Token，可完美收拢全模型的冗余注意力分值”的论文核心创见 。