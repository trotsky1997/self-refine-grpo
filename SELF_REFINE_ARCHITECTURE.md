# Self-Refine Architecture

## Overview

Multi-round self-refinement with three specialized sub-agents:

```
┌─────────────────────────────────────────────────────────────┐
│                    SelfRefineAgent                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1️⃣ Answer Agent    → Generate initial/refined answers       │
│                                                               │
│  2️⃣ Critique Agent  → Evaluate and critique answers          │
│                                                               │
│  3️⃣ Refine Agent    → Improve based on critique              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Round Refinement Flow

```
Round 0:  [Question] → Answer Agent → [Answer₀]
                                           ↓
Round 1:  [Answer₀] → Critique Agent → [Critique₁]
                         ↓
          [Question + Critique₁] → Refine Agent → [Answer₁]
                                           ↓
Round 2:  [Answer₁] → Critique Agent → [Critique₂]
                         ↓
          [Question + Critique₂] → Refine Agent → [Answer₂]
                                           ↓
          ... (continue for max_refine_rounds) ...
```

## Agent Specifications

### 1. Answer Agent
- **Input**: Question + optional context
- **Output**: `<think>...</think>` + `<answer>...</answer>`
- **System Prompt**: Standard problem-solving prompt
- **Usage**: 
  - Round 0: Generate initial answer
  - Round N: Generate refined answer based on critique

### 2. Critique Agent
- **Input**: Question + Answer to evaluate
- **Output**: Constructive feedback
- **System Prompt**: Critical reviewer role
- **Focus**:
  - ✅ Correctness of reasoning and calculations
  - ✅ Completeness of the solution
  - ✅ Clarity and organization
  - ✅ Common mistakes or edge cases

### 3. Refine Agent
- **Input**: Question + Previous answer + Critique
- **Output**: `<think>...</think>` + `<answer>...</answer>`
- **System Prompt**: Problem solver with feedback
- **Strategy**: Address critique points while maintaining correct aspects

## Configuration

### Basic Usage (Single Round)

```python
trainer = GRPOTrainer(
    enable_self_refine=True,
    use_critique=True,
    max_refine_rounds=1,  # Default: 1 round
    refine_log_file="self_refine_log.jsonl",
    ...
)
```

**Flow**: Question → Answer₀ → Critique → Answer₁

### Multi-Round Refinement

```python
trainer = GRPOTrainer(
    enable_self_refine=True,
    use_critique=True,
    max_refine_rounds=3,  # 3 rounds of refinement
    refine_log_file="self_refine_log.jsonl",
    ...
)
```

**Flow**: Question → Answer₀ → (Critique₁ → Answer₁) × 3

### Disable Critique (Direct Refine)

```python
trainer = GRPOTrainer(
    enable_self_refine=True,
    use_critique=False,  # Skip critique step
    max_refine_rounds=1,
    ...
)
```

**Flow**: Question → Answer₀ → Answer₁ (no critique)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_self_refine` | bool | `False` | Enable self-refinement |
| `use_critique` | bool | `True` | Use critique agent before refining |
| `max_refine_rounds` | int | `1` | Number of refinement rounds |
| `refine_log_file` | str | `"self_refine_log.jsonl"` | Path to log file |
| `return_both_responses` | bool | `False` | Return original + refined (doubles batch) |

## Logging

Each refinement is logged to `self_refine_log.jsonl`:

```json
{
  "prompt": "Question text",
  "original_answer": "Initial answer",
  "original_correct": false,
  "critiques": ["Critique for round 1", "Critique for round 2"],
  "refined_answers": ["Answer after round 1", "Answer after round 2"],
  "final_correct": true,
  "rounds": 2,
  "improved": true
}
```

## Statistics

After each batch, the trainer prints detailed statistics:

```
Batch 5: Accuracy@t0=0.50 → Accuracy@t1=0.75 (Δ +0.25)
  Transitions: i→c=2, i→i=2, c→c=4, c→i=0
```

**Legend**:
- `Accuracy@t0`: Initial accuracy (before refinement)
- `Accuracy@t1`: Final accuracy (after all rounds)
- `i→c`: Incorrect → Correct
- `i→i`: Incorrect → Incorrect
- `c→c`: Correct → Correct
- `c→i`: Correct → Incorrect

## Implementation Details

### Class: `SelfRefineAgent`

Located in `trainer.py`, this class encapsulates all refinement logic:

```python
class SelfRefineAgent:
    def __init__(self, trainer, max_refine_rounds: int = 1):
        self.trainer = trainer
        self.max_refine_rounds = max_refine_rounds
        # Initialize system prompts for each agent
    
    def answer_agent(self, batch, context="initial"):
        # Generate answers using model
    
    def critique_agent(self, prompts, answers, round_idx=0):
        # Generate critiques for answers
    
    def refine_agent(self, prompts, answers, critiques, round_idx=0):
        # Generate refined answers based on critiques
    
    def run_multi_round_refine(self, batch, incorrect_mask):
        # Orchestrate multiple rounds of refinement
```

### Integration

The agent is automatically instantiated in `GRPOTrainer.__init__` when `enable_self_refine=True`:

```python
if self.enable_self_refine:
    self.refine_agent = SelfRefineAgent(self, max_refine_rounds=max_refine_rounds)
```

## Examples

### Example 1: Conservative (1 round)
Good for: Fast iteration, limited compute

```python
max_refine_rounds=1
```

### Example 2: Moderate (2 rounds)
Good for: Balance between quality and speed

```python
max_refine_rounds=2
```

### Example 3: Aggressive (3+ rounds)
Good for: Maximum quality, sufficient compute

```python
max_refine_rounds=3
```

## Benefits

✅ **Modular Design**: Each agent has a clear, single responsibility

✅ **Configurable**: Easily adjust refinement depth

✅ **Scalable**: Supports arbitrary number of rounds

✅ **Observable**: Detailed logging for each refinement step

✅ **GRPO Compatible**: Generates multiple responses for policy optimization

## Future Enhancements

🔄 **Adaptive Rounds**: Stop early if answer converges

🔄 **Confidence Thresholding**: Refine only low-confidence answers

🔄 **Ensemble Refinement**: Generate multiple critiques and vote

🔄 **Curriculum Learning**: Start with more rounds, gradually reduce

---

**Note**: The current implementation in `_generate_and_score_completions` uses the legacy two-stage logic. Full integration of `SelfRefineAgent` will be completed in a future update.
