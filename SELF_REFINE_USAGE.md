# Self-Refine Agent Usage Guide

## 架构概览

Self-refine 功能现已模块化为三个独立的子 agent：

```
SelfRefineAgent (self_refine_agent.py)
├── Answer Agent    - 生成初始/改进答案
├── Critique Agent  - 评估并批评当前答案
└── Refine Agent    - 基于批评改进答案
```

## 文件结构

- **`self_refine_agent.py`** - Self-refine agent 类定义
  - 独立的三个子 agent 方法
  - 支持单轮和多轮改进接口
  
- **`trainer.py`** - GRPO trainer（已简化）
  - 导入并使用 `SelfRefineAgent`
  - 在 `_generate_and_score_completions` 中调用 agent 方法

- **`grpo_vlm.py`** - 配置和启动
  - 设置 `max_refine_rounds` 参数

## 使用方式

### 1. 基础配置（单轮改进）

```python
trainer = GRPOTrainer(
    ...,
    enable_self_refine=True,      # 启用 self-refine
    use_critique=True,             # 启用两阶段 refine（critique + refine）
    max_refine_rounds=1,           # 单轮改进
)
```

### 2. 多轮改进（未来扩展）

```python
trainer = GRPOTrainer(
    ...,
    enable_self_refine=True,
    use_critique=True,
    max_refine_rounds=3,  # 三轮迭代改进
)
```

## Agent 工作流程

### Stage 1: Critique Agent

```
不正确样本 → Critique Agent → 生成批评
```

**输入**：
- 原始问题 prompts
- 不正确的答案

**输出**：
- 结构化批评（评估推理、计算、完整性）

### Stage 2: Refine Agent

```
原始问题 + 原始答案 + 批评 → Refine Agent → 生成改进答案
```

**输入**：
- 原始问题 prompts
- 不正确的答案
- 批评内容

**输出**：
- 改进后的答案

## 提示词模板

### Critique Agent System Prompt

```
You are a critical reviewer. Evaluate the given solution and provide constructive feedback.

Focus on:
- Correctness of reasoning and calculations
- Completeness of the solution
- Clarity and organization
- Common mistakes or edge cases

Format your critique in <think></think> tags for internal analysis, 
followed by clear, actionable feedback.
```

### Refine Agent System Prompt

```
You are a problem solver. Given a question and feedback on a previous attempt, 
provide an improved solution.

Use the standard format:
<think>Your unstructured internal reasoning</think>
<answer>Clear, well-formatted solution with final answer in \boxed{}</answer>
```

## 统一接口设计 ⭐

### 核心原则

**trainer.py 只调用一次，agent 内部处理所有工作流程！**

这样设计的好处：
1. **完全黑盒** - Agent 内部处理一切，trainer 不知道任何细节
2. **易扩展** - 未来添加其他 agent 只需实现相同接口
3. **极简** - trainer 代码只需传入 inputs，接收 outputs

### 统一接口

**唯一调用点 - 一次搞定所有！**
```python
# ⭐ ULTIMATE UNIFIED INTERFACE ⭐
# Single call - agent handles EVERYTHING:
# - Critique generation
# - Refine prompt building  
# - Regeneration
# - Evaluation
# - Logging
refined_result = agent.refine_batch(
    inputs=inputs,
    original_result=result,
    original_prompts=prompts,
    answers=completions_text,
    solutions=solutions,
    correctness_mask=correctness_tensor,
    batch_counter=batch_counter
)
```

**Agent 内部自动完成**：
- ✅ Critique 生成
- ✅ Refine prompt 构建
- ✅ Prompt 展开到 full batch
- ✅ 调用 generation（内部）
- ✅ 解码 completions（内部）
- ✅ 评估统计（内部）
- ✅ 日志记录（内部）
- ✅ 结果打印（内部）
- ✅ 返回 refined result

## 代码改进

### 之前（暴露所有内部细节）

```python
# trainer.py: 300+ 行手动处理所有逻辑

# 1. 打印统计
print(f"Batch size: {batch_size}")
print(f"Accuracy@t1: {batch_acc_t1:.2f}%")
# ... 更多统计打印

# 2. 手动构建 critique prompts
for idx, inp in enumerate(inputs):
    if needs_refine_mask[idx]:
        critique_prompt = []
        for i, msg in enumerate(original_prompt):
            # ... 100+ 行构建逻辑

# 3. 生成 critique
critique_result = super()._generate_and_score_completions(...)

# 4. 手动构建 refine prompts
for idx, inp in enumerate(inputs):
    if needs_refine_mask[idx]:
        refine_prompt = []
        # ... 100+ 行构建逻辑

# 5. 手动评估
for idx in range(len(completions_text)):
    is_correct = self._check_if_correct(...)
    # ... 统计 transitions

# 6. 手动打印结果
print(f"Accuracy@t2: {batch_acc_t2:.2f}%")
# ... 更多输出

# 7. 手动记录日志
for idx in range(len(completions_text)):
    log_entry = {...}
    self._log_refine_sample(log_entry)
```

### 之后（完全黑盒）

```python
# trainer.py: 仅 ~5 行！🎉

# ⭐ 一次调用，搞定所有！
refined_result = self.refine_agent.refine_batch(
    inputs=inputs,
    original_result=result,
    original_prompts=prompts,
    answers=completions_text,
    solutions=solutions,
    correctness_mask=correctness_tensor,
    batch_counter=self.batch_counter
)

# Done! 🎉
# Agent 内部自动处理：
# - Critique 生成
# - Refine prompts 构建
# - Prompt 展开到 full batch
# - 调用 generation
# - 解码 completions
# - 评估统计（accuracy、transitions）
# - 结果打印（格式化输出）
# - 日志记录（JSONL 文件）
# - 返回 refined result

# Trainer 完全不知道内部发生了什么！✨
```

**代码行数减少**：300+ 行 → 5 行（**-98%！**）

## 日志输出

训练时会看到由 agent 控制的简洁输出：

```
[Self-Refine Agent] Processing 4 incorrect samples using Self-Refine Agent (Critique + Refine)

📊 Self-Refine Agent Evaluation (Batch 1):
  Accuracy@t1: 4/8 = 50.00%
  Accuracy@t2: 6/8 = 75.00%
  Δ(batch):    +25.00% 📈

  Transitions:
    i→c:  2  |  i→i:  2
    c→c:  4  |  c→i:  0

  Example trace:
    Original: To solve this problem...
    Critique: The reasoning is correct but...
    Refined:  Using the corrected approach...
    Result: ✓ Correct

  ✓ Logged 8 samples to self_refine_log.jsonl
```

**关键改进**：
- ✅ 所有输出由 agent 控制，不暴露给 trainer
- ✅ 格式统一，易于解析
- ✅ 包含完整的评估统计
- ✅ 自动记录到 JSONL 文件

## 扩展多轮改进

当前实现支持单轮改进。要启用多轮迭代：

1. 在 `grpo_vlm.py` 中设置 `max_refine_rounds > 1`
2. Agent 会自动迭代多轮（critique → refine → critique → refine...）
3. 每轮结果都会记录到 `refine_history`

```python
# 未来：多轮改进会自动执行
for round in range(max_refine_rounds):
    critiques = critique_agent(...)
    refined_answers = refine_agent(..., critiques, round_idx=round)
    # 评估并决定是否继续
```

## 扩展其他采样 Agent

统一接口设计使得添加新的采样 agent 变得容易：

### 步骤 1: 创建新 Agent 类

```python
# my_custom_agent.py
class MyCustomSamplingAgent:
    """自定义采样 agent"""
    
    def __init__(self, trainer, **kwargs):
        self.trainer = trainer
        # ... 初始化参数
    
    # ⭐ 实现统一接口（一次调用）
    def refine_batch(
        self,
        inputs,
        original_result,
        original_prompts,
        answers,
        solutions,
        correctness_mask,
        batch_counter=0
    ):
        """
        统一接口：处理整个 batch，返回 refined result
        
        Agent 内部自主处理：
        - 生成、评估、日志
        """
        # 1. 提取 incorrect samples（内部）
        incorrect_indices = torch.where(~correctness_mask)[0].tolist()
        if not incorrect_indices:
            return None
        
        # 2. 你的自定义采样逻辑（内部）
        # - 可以用强化学习
        # - 可以用 beam search
        # - 可以用多轮迭代
        refined_prompts = self._your_custom_sampling(...)
        
        # 3. 展开到 full batch（内部）
        refine_inputs = self._expand_to_full_batch(...)
        
        # 4. 调用生成（内部）
        refined_result = _GRPOTrainer._generate_and_score_completions(
            self.trainer, refine_inputs
        )
        
        # 5. 评估和日志（内部）
        self._evaluate_and_log(...)
        
        # 6. 返回结果
        return refined_result
```

### 步骤 2: 在 trainer.py 中切换 Agent

```python
# trainer.py __init__
if self.enable_self_refine:
    if agent_type == "self_refine":
        self.refine_agent = SelfRefineAgent(self, max_refine_rounds)
    elif agent_type == "custom":
        self.refine_agent = MyCustomSamplingAgent(self, **kwargs)
    # ... 更多 agent 类型
```

### 步骤 3: 无需修改调用代码

```python
# trainer.py _generate_and_score_completions
# 无论什么 agent，调用方式完全相同！⭐
refined_result = self.refine_agent.refine_batch(
    inputs=inputs,
    original_result=result,
    original_prompts=prompts,
    answers=completions_text,
    solutions=solutions,
    correctness_mask=correctness_tensor,
    batch_counter=self.batch_counter
)
```

**关键点**：
- ✅ 新 agent 只需实现 `refine_batch()` 接口（一次调用，返回结果）
- ✅ trainer.py **完全无需修改**，只是传入数据，接收结果
- ✅ Agent 内部自主决定：生成、评估、日志
- ✅ 可以自由切换不同 agent
- ✅ 保持代码极简（5 行）和可维护性

## 优势总结

| 指标 | 之前 | 之后 | 改进 |
|------|------|------|------|
| **trainer.py 代码行数** | 300+ 行 | **5 行** | **-98%** 🎉 |
| **trainer 需要的调用次数** | 手动构建 prompts + 生成 + 解码 + 评估 + 日志 | **仅 1 次** | **极简** |
| **暴露的内部细节** | critique/refine/评估/日志/生成/解码 | **无** | **完全黑盒** ✨ |
| **trainer 是否知道 agent 做了什么** | 是（手动处理所有步骤） | **否** | **完全封装** |
| **日志控制** | trainer | agent | **agent 自主** |
| **评估控制** | trainer | agent | **agent 自主** |
| **生成控制** | trainer | agent | **agent 自主** |
| **接口复杂度** | 多步骤手动调用 | **单次调用** | **极简** |

### 核心优势

1. **完全黑盒** ⭐⭐⭐ - Trainer 完全不知道 agent 内部做了什么，只是传入数据，等待结果
2. **极简接口** - Trainer 只需 **1 次调用**，代码量减少 **98%**
3. **可扩展** - 轻松添加新的 agent，只需实现 `refine_batch()` 接口
4. **可测试** - Agent 可独立测试，trainer 无需关心任何细节
5. **灵活** - 支持单轮/多轮，有/无 critique，agent 内部决定
6. **可插拔** - 不同 agent 可自由切换，无需修改 trainer
7. **自主控制** - Agent 决定一切：生成、评估、日志、打印
8. **真正的代理模式** - Trainer 只是委托任务，agent 完全自主执行

## 文件清理

已删除冗余文件：
- `benchmark_gspo.py`
- `check_optimizations.py`
- `test_lora_triton.py`
- `test_triton_integration.py`
- `test.sh`
- `agentic-sampler.py`

保留核心文件：
- `self_refine_agent.py` ⭐ 新增：Agent 封装
- `sampler.py` ⭐ 新增：Sampler 解耦
- `trainer.py` ✅ 简化
- `grpo_vlm.py` ✅ 配置
- `launch.sh` ✅ 启动

详细架构文档：
- `SELF_REFINE_ARCHITECTURE.md` - Agent 架构
- `SAMPLER_ARCHITECTURE.md` - Sampler 架构
