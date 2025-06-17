import os

def compute_score(solution_str):
    """
    Computes comprehensive score for model response.
    Args:
        solution_str: Raw model response string
        ground_truth: ground truth data
        answer_reward: Points awarded/deducted for answer correctness
    Returns:
        Total score (sum of format and answer rewards)
    """
    
    match = re.search(r'<answer>(.*?)</answer>', output)
    if match:
        math_problem = match.group(1)
        if re.search(r'[a-zA-Z]', math_problem) and len(math_problem) > 30: 
        # 只有数字的扔掉，题目太短的扔掉30是根据" x = -xx, y = -xx, z = -xx, "这个赋值表达式算来的
            print(f"A valid question. {math_problem}")
            return 1.0
        else:
            print(f"Only numbers in question or the question is too short. {math_problem}")
            return 0.0
            
    else:
        print("No question found in the output string.")
        return 0.0