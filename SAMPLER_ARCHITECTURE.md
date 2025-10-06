# Sampler Architecture

## 概述

Sampler 模块提供了统一的采样（generation）接口，将生成逻辑从 trainer 和 agent 中解耦。

## 核心设计

### 设计原则

1. **统一接口** - 所有 sampler 提供相同的接口
2. **独立解耦** - 生成逻辑与训练/agent 逻辑完全分离
3. **可扩展** - 轻松添加新的 sampler 实现（vLLM、HuggingFace、Beam Search等）
4. **易测试** - Sampler 可独立测试，不依赖完整的 trainer

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        GRPOTrainer                          │
│                                                             │
│  ┌───────────────┐                  ┌──────────────────┐   │
│  │   Training    │                  │  SelfRefineAgent │   │
│  │    Logic      │                  │                  │   │
│  └───────┬───────┘                  └─────────┬────────┘   │
│          │                                    │            │
│          │                                    │            │
│          └──────────────┬─────────────────────┘            │
│                         │                                  │
│                         ▼                                  │
│                  ┌─────────────┐                           │
│                  │   Sampler   │  ◄──── Unified Interface │
│                  └─────────────┘                           │
│                         │                                  │
└─────────────────────────┼──────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │   Sampler Implementations   │
            ├─────────────────────────────┤
            │ • VLLMSampler               │
            │ • HuggingFaceSampler        │
            │ • BeamSearchSampler         │
            │ • CustomSampler             │
            └─────────────────────────────┘
```

## Sampler 接口

### BaseSampler

所有 sampler 的基类。

```python
class BaseSampler:
    def __init__(self, trainer):
        self.trainer = trainer
    
    def generate_and_score(self, inputs: List[Dict]) -> Dict:
        """
        高级接口：从结构化 inputs 生成
        
        Args:
            inputs: List of input dicts, each containing:
                - prompt: conversation history
                - pixel_values: visual inputs (optional)
                - ... other fields
        
        Returns:
            Dict with completion_ids, per_token_logps, etc.
        """
        raise NotImplementedError
```

### VLLMSampler

默认 sampler，使用 vLLM 进行高吞吐量生成。

```python
class VLLMSampler(BaseSampler):
    def generate_and_score(self, inputs):
        """使用 vLLM 生成（通过 TRL 的 _generate_and_score_completions）"""
        from trl import GRPOTrainer as _GRPOTrainer
        return _GRPOTrainer._generate_and_score_completions(self.trainer, inputs)
    
    def generate_from_prompts(self, prompts, images=None):
        """
        低级接口：从原始 prompts 生成
        
        Args:
            prompts: List of conversation prompts or strings
            images: Optional list of images
        
        Returns:
            Dict with completion_ids, etc.
        """
        from trl import GRPOTrainer as _GRPOTrainer
        return _GRPOTrainer._generate_single_turn(self.trainer, prompts, images)
```

## 使用示例

### Trainer 中使用

```python
# trainer.py
class GRPOTrainer(_GRPOTrainer):
    def __init__(self, ...):
        # 创建 sampler
        from sampler import create_sampler
        self.sampler = create_sampler(self, sampler_type="vllm")
        
        # 创建 agent（agent 会使用 trainer.sampler）
        self.refine_agent = SelfRefineAgent(self)
    
    def _generate_and_score_completions(self, inputs):
        # 使用 sampler 生成
        result = self.sampler.generate_and_score(inputs)
        
        # 自定义逻辑（如 self-refine）
        if self.enable_self_refine:
            refined_result = self.refine_agent.refine_batch(...)
        
        return result
```

### Agent 中使用

```python
# self_refine_agent.py
class SelfRefineAgent:
    def critique_agent(self, prompts, answers):
        # 使用 sampler 生成 critiques
        critique_result = self.trainer.sampler.generate_from_prompts(
            critique_prompts, 
            images=None
        )
        return critique_texts
    
    def refine_batch(self, inputs, ...):
        # 构建 refined prompts
        refined_prompts = self._build_refined_prompts(...)
        
        # 使用 sampler 生成 refined answers
        refined_result = self.trainer.sampler.generate_and_score(refine_inputs)
        
        return refined_result
```

## 扩展新 Sampler

### 步骤 1: 创建 Sampler 类

```python
# sampler.py
class MyCustomSampler(BaseSampler):
    def __init__(self, trainer, **kwargs):
        super().__init__(trainer)
        self.my_param = kwargs.get('my_param', default_value)
    
    def generate_and_score(self, inputs):
        # 你的自定义生成逻辑
        # 例如：使用 beam search、强化学习等
        ...
        return {
            'completion_ids': ...,
            'completion_mask': ...,
            'policy_per_token_logps': ...,
            ...
        }
```

### 步骤 2: 注册 Sampler

```python
# sampler.py
def create_sampler(trainer, sampler_type="vllm", **kwargs):
    samplers = {
        "vllm": VLLMSampler,
        "huggingface": HuggingFaceSampler,
        "beam_search": BeamSearchSampler,
        "my_custom": MyCustomSampler,  # ✅ 添加新 sampler
    }
    return samplers[sampler_type](trainer, **kwargs)
```

### 步骤 3: 使用新 Sampler

```python
# grpo_vlm.py or trainer.py
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    # ... other args
)

# 切换到自定义 sampler
trainer.sampler = create_sampler(trainer, sampler_type="my_custom", my_param=123)
```

## 优势总结

| 指标 | 之前 | 之后 | 改进 |
|------|------|------|------|
| **调用方式** | 直接调用 `super()._generate_and_score_completions()` | 调用 `self.sampler.generate_and_score()` | **统一接口** |
| **耦合度** | 强耦合到 TRL 实现 | 完全解耦 | **独立模块** |
| **可扩展性** | 需修改 trainer/agent | 只需添加新 sampler 类 | **易扩展** |
| **可测试性** | 依赖完整 trainer | Sampler 可独立测试 | **易测试** |
| **切换成本** | 需修改多处代码 | 只需切换一行 | **极低** |

### 核心优势

1. **完全解耦** ⭐⭐⭐
   - Trainer 不知道具体的生成实现（vLLM/HF/Beam Search）
   - Agent 不依赖 TRL 的内部方法
   - 生成逻辑独立可测试

2. **统一接口** 
   - 所有 sampler 提供相同的 API
   - Trainer 和 agent 代码无需修改即可切换 sampler

3. **易扩展**
   - 添加新 sampler 只需实现 `BaseSampler` 接口
   - 无需修改现有代码

4. **灵活切换**
   - 开发时用 HuggingFace（慢但灵活）
   - 生产时用 vLLM（快速高吞吐）
   - 研究时用 Beam Search/自定义算法

## 文件结构

```
/home/diz/cvpr/
├── sampler.py                      # ⭐ 新增：Sampler 模块
│   ├── BaseSampler                 # 基类
│   ├── VLLMSampler                 # vLLM 实现（默认）
│   ├── HuggingFaceSampler          # HF 实现（TODO）
│   ├── BeamSearchSampler           # Beam search（TODO）
│   └── create_sampler()            # 工厂函数
├── trainer.py                      # ✅ 使用 sampler
├── self_refine_agent.py            # ✅ 使用 sampler
├── grpo_vlm.py                     # ✅ 配置
└── launch.sh                       # ✅ 启动
```

## 设计模式

这个架构使用了以下设计模式：

1. **策略模式 (Strategy Pattern)** - `BaseSampler` 定义统一接口，不同实现可互换
2. **工厂模式 (Factory Pattern)** - `create_sampler()` 根据类型创建实例
3. **依赖注入 (Dependency Injection)** - Trainer/Agent 通过 `self.sampler` 注入依赖

这使得系统：
- **松耦合** - 各模块独立
- **高内聚** - 每个模块职责清晰
- **易维护** - 修改生成逻辑不影响其他代码
- **可测试** - 可以 mock sampler 进行单元测试

