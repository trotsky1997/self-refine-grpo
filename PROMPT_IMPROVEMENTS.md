# Self-Refine Prompt Improvements

## Summary

Improved the efficiency of the critic and improver agent prompts to provide more targeted, actionable feedback.

## Key Changes

### 1. Critic Agent Improvements

**Before:**
- Only saw the answer (without `<think>` tags)
- No access to ground truth
- Generic evaluation questions
- Vague instructions

**After:**
- Sees the FULL solution including `<think>` tags (complete reasoning process)
- Has access to the CORRECT ANSWER for comparison
- Focused on identifying:
  - WHERE the reasoning went wrong
  - WHAT misconceptions led to errors
  - Specific steps/calculations that failed
- Requires brief, focused critique (2-4 sentences)
- Actionable suggestions (2-3 specific points)

**Benefits:**
- Critic can provide targeted feedback by comparing against ground truth
- Sees the full reasoning chain to identify where logic breaks down
- More precise error identification
- Shorter, more actionable critiques

### 2. Improver Agent Improvements

**Before:**
- Saw "a reference solution" and "key observations"
- Vague instruction to "provide your solution"
- Risk of just patching/tweaking the previous attempt

**After:**
- Only sees the CRITIQUE (not the previous solution)
- Explicitly instructed to solve FROM SCRATCH
- Clear warning: "Do NOT try to patch the previous solution"
- Guidance on what to apply:
  - Where the reasoning went wrong
  - What approach would be correct
  - Key concepts needing attention

**Benefits:**
- Forces fresh reasoning instead of incremental patching
- Learns from mistakes without anchoring to bad solutions
- Applies insights gained from critique
- More likely to find correct approach independently

## Implementation Details

### Critic Prompt Structure

```
System: You are an expert critic with access to:
1. Original problem
2. Student's full solution (with reasoning)
3. Correct answer

Task: Compare, identify errors, provide 2-3 actionable suggestions

User: [problem]

=== STUDENT'S SOLUTION (including reasoning) ===
[full completion with <think> tags]

=== CORRECT ANSWER ===
[ground truth]

Analyze: Where did they go wrong? What should they focus on?
```

### Improver Prompt Structure

```
System: [Original system prompt - unchanged]

User: [problem]

=== PREVIOUS ATTEMPT ANALYSIS ===
Your earlier attempt had issues. Expert feedback:
[critique]

=== YOUR TASK ===
Absorb the feedback, then solve FROM SCRATCH.
Do NOT patch - start fresh with insights gained.
Apply what you learned about:
- Where reasoning went wrong
- What approach is correct
- Key concepts needing attention

Now solve step by step.
```

## Expected Impact

1. **Higher i→c transition rate**: More incorrect answers becoming correct
2. **More efficient learning**: Targeted feedback vs generic critique
3. **Better reasoning quality**: Fresh solutions vs patched attempts
4. **Reduced token usage**: Shorter, focused critiques
5. **Improved sample efficiency**: Each refine round more effective

## Files Modified

- `trainer.py`:
  - Lines 470-525: Critic prompt generation
  - Lines 548-605: Improver prompt generation

