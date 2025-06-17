import os
from verl.utils.reward_score.math_utils import is_equal, solution2answer

def compute_format_score(solution_str: str):
    answer_text = solution2answer(solution_str)
    if not answer_text:
        print('thinking format wrong, return') # 得不到答案，不鼓励format
        return 0.0

def compute_score(solution_str: str, ground_truth: str):
    """
    Computes comprehensive score for model response.
    Args:
        solution_str: Raw model response string
        ground_truth: ground truth data
        answer_reward: Points awarded/deducted for answer correctness
    Returns:
        Total score (sum of format and answer rewards)
    """
    
    answer_text = solution2answer(solution_str)
    ground_truth = solution2answer(ground_truth)
    
    print("\n\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))
    print(f"[Ground Truth]\n{ground_truth}\n")
    print(f"[Model Response]\n{solution_str}\n")
    print(f"[Extracted Answer]\n{answer_text}\n")
        
    if not answer_text:
        print('thinking format wrong, return') # 得不到答案，不鼓励format
        return 0.0
    
    score = float(is_equal(ground_truth, answer_text))
    print(f"[Is Correct?]\n{score}")
    
    print("=" * 80 + "\n")

    return score