# qwen3_vl/reward_functions.py
"""
Reward functions for GRPO/GSPO reinforcement learning training
Used to score model completions based on formatting and correctness
"""

import re
from typing import List


# Define the delimiter variables for clarity and easy modification
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def formatting_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function for checking proper formatting of model outputs
    
    Checks for:
    - Exactly one reasoning section (<REASONING>...</REASONING>)
    - Exactly one solution section (<SOLUTION>...</SOLUTION>)
    - Penalizes excessive "addCriterion" tokens and newlines (Qwen2.5-VL quirk)
    
    Args:
        completions: List of model-generated text completions
        **kwargs: Additional arguments (unused)
    
    Returns:
        List of scores, one per completion (range: -2.0 to +2.0)
    """
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    scores = []
    for completion in completions:
        score = 0.0

        # Check reasoning section (must have exactly 1)
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0

        # Check solution section (must have exactly 1)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(answer_matches) == 1:
            score += 1.0

        # Fix up addCriterion issues
        # See https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks
        # Penalize excessive addCriterion and newlines
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0

        scores.append(score)

    return scores


def correctness_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward function for checking correctness of predicted answers
    
    Extracts the answer from <SOLUTION>...</SOLUTION> tags and compares
    with ground truth. Prints first example for debugging.
    
    Args:
        prompts: List of input prompts (for debugging)
        completions: List of model-generated text completions
        answer: List of ground truth answers
        **kwargs: Additional arguments (unused)
    
    Returns:
        List of scores: 2.0 for correct, 0.0 for incorrect
    """
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    # Extract answers from completions
    responses = [
        re.findall(answer_pattern, completion, re.DOTALL) for completion in completions
    ]

    # Print first example for debugging
    q = prompts[0]
    print("-" * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")

    # Score: 2.0 if exactly one answer extracted and matches ground truth
    return [
        2.0 if len(r) == 1 and a == r[0].replace("\n", "") else 0.0
        for r, a in zip(responses, answer)
    ]
