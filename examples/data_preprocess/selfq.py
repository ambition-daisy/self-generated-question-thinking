import os
import datasets
import argparse
from datasets import Dataset

def make_map_fn(split):

    def process_init_train_math(example, idx):
        question = example['0']['value']
        chat = [{'role':'user', 'content': f'Now I ask you a question and you solve it. You should first thinks about the reasoning process in the mind and then provides me with the answer at the end. You must put your answer inside <answer> </answer> tags. And your final answer must be enclosed by \\boxed{{}}, i.e., <answer> \\boxed{{your answer}} </answer>. Here is my question: {question}'}]

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('/root/Qwen2.5-7B')
        # prompt_with_chat_template = tokenizer.apply_chat_template(chat, tokenize=False)
        # input(prompt_with_chat_template)

        data = {
            "data_source": 'train_math',
            "prompt": chat,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": None,
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }

        return data
    
    def process_aug(example, idx):
        question = example['data']
        chat = [{'role':'user', 'content': f'Now, I ask you to generate a math problem. Please carefully think through the reasoning process of how to create a challenging and high-quality problem, and then present the final problem at the end. You must put your generated problem inside <answer> </answer> tags, i.e., <answer> your generated math problem </answer>. Note that you should ONLY present the final problem enclosed in the answer tags, without any solutions or explanations, so that my students can independently solve it. Here are the problem requirements: {question}.'}]

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('/root/Qwen2.5-7B')
        # prompt_with_chat_template = tokenizer.apply_chat_template(chat, tokenize=False)
        # input(prompt_with_chat_template)

        data = {
            "data_source": 'train_question',
            "prompt": chat,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": None,
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data
    
    def process_aime(example, idx):
        question = example['prompt'][0]['value']
        chat = [{'role':'user', 'content': f'Now I ask you a question, and you solve it. You should first thinks about the reasoning process in the mind and then provides me with the answer at the end. You must put your answer inside <answer> </answer> tags. And your final answer must be enclosed by \\boxed{{}}, i.e., <answer> \\boxed{{your answer}} </answer>. Here is my question: {question}'}]
        answer = example["final_answer"]
        data = {
            "data_source": 'aime',
            "prompt": chat,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data
    
    def process_gpqa(example, idx):
        question = example['prompt'][0]['value']
        chat = [{'role':'user', 'content': f'Now I ask you a question, and you solve it. You should first thinks about the reasoning process in the mind and then provides me with the answer at the end. You must put your answer choice (a single capital letter) inside <answer> </answer> tags. And your final answer must be enclosed by \\boxed{{}}, i.e., <answer> \\boxed{{your answer}} </answer>. Here is my question: {question}'}]
        answer = example["final_answer"]
        data = {
            "data_source": 'gpqa',
            "prompt": chat,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data

    def process_math(example, idx):
        question = example['prompt'][0]['value']
        chat = [{'role':'user', 'content': f'Now I ask you a question, and you solve it. You should first thinks about the reasoning process in the mind and then provides me with the answer at the end. You must put your answer inside <answer> </answer> tags. And your final answer must be enclosed by \\boxed{{}}, i.e., <answer> \\boxed{{your answer}} </answer>. Here is my question: {question}'}]
        answer = example["final_answer"]
        data = {
            "data_source": 'math500',
            "prompt": chat,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data


    if split == 'train':
        return process_init_train_math
    elif split == 'aime':
        return process_aime
    elif split == 'gpqa':
        return process_gpqa
    elif split == 'math500':
        return process_math
    elif split == 'aug':
        return process_aug

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/mnt/wx_feature/home/anglv/verl/datasets/selfq')
    args = parser.parse_args()
    
    local_dir = args.local_dir

    # train set
    train_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Think/datasets/orz/orz_math_57k_collected.json',
    })['data']
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    sampled_train_dataset = train_dataset.shuffle(seed=42).select(range(200))
    sampled_train_dataset.to_parquet(os.path.join(local_dir, 'init_train_200.parquet'))
    
    
    # test set
    math_test_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Open-Reasoner-Zero/data/eval_data/math500.json',
    })['data']
    math_test_dataset = math_test_dataset.map(function=make_map_fn('math500'), with_indices=True)
    math_test_dataset.to_parquet(os.path.join(local_dir, 'math_test.parquet'))
    
    aime_test_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Open-Reasoner-Zero/data/eval_data/aime2024.json',
    })['data']
    aime_test_dataset = aime_test_dataset.map(function=make_map_fn('aime'), with_indices=True)
    aime_test_dataset.to_parquet(os.path.join(local_dir, 'aime_test.parquet'))
    
    
    gpqa_test_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Open-Reasoner-Zero/data/eval_data/gpqa_diamond.json',
    })['data']
    gpqa_test_dataset = gpqa_test_dataset.map(function=make_map_fn('gpqa'), with_indices=True)
    gpqa_test_dataset.to_parquet(os.path.join(local_dir, 'gpqa_test.parquet'))
    
    # augment set
    aug_dict = {'data':[
        # Algebra
        "Create an algebra problem that involves solving a system of linear equations using both substitution and elimination methods. The system should have three variables and lead to a non-integer solution.",
        "Design an algebra problem where the goal is to solve a polynomial equation of degree 4. The roots should include both real and complex numbers, and the problem should require factoring and the use of the Rational Root Theorem.",
        "Generate an algebra question that involves simplifying and solving a rational equation with fractional expressions. The denominators should contain polynomials, and the solution should require checking for extraneous solutions.",
        "Create a word problem that requires setting up and solving a quadratic equation to model a real-world situation, such as projectile motion or the area of a rectangle. The problem should have both a positive and a negative solution.",
        "Design an algebraic expression problem where students need to simplify an expression involving multiple exponents and logarithms. The solution should include the application of the power rule and logarithmic properties.",
        "Generate an equation-solving problem involving absolute value. The absolute value expression should be set equal to a negative number, and the problem should require students to understand why there is no solution.",
        "Create an algebra problem involving a geometric sequence. The task should be to find the nth term of the sequence and calculate the sum of the first 10 terms, using the formula for the sum of a geometric series.",
        "Design a problem where the goal is to find the roots of a quadratic function using the quadratic formula. Include a scenario where the discriminant is zero, indicating a double root.",
        "Generate a problem that requires factoring a cubic polynomial and solving for the roots. The cubic should have one real root and two complex roots, and the solution should involve the use of synthetic division or long division.",
        "Create an algebra question involving a linear inequality with absolute values. The problem should require students to express the solution as a compound inequality and graph the solution on a number line.",
        "Create a problem that requires solving a system of non-linear equations involving quadratic and cubic functions. Use both substitution and graphical methods to find the solutions.",
        "Design an algebra problem where the goal is to solve a rational equation that involves square roots. The problem should require simplifying the expression and checking for extraneous solutions.",
        "Generate a problem that asks students to solve a quadratic equation by completing the square. The equation should have both a real and an imaginary root.",
        "Design a problem that involves solving a system of quadratic equations. The system should have one real solution and two imaginary solutions.",
        "Create an algebraic expression problem involving both logarithms and trigonometric functions. The task should be to simplify the expression using logarithmic identities and trigonometric formulas.",
        "Design a word problem involving the cost of materials for a construction project. Students should model the problem using a quadratic equation and find the break-even point.",
        "Generate a problem that involves finding the sum and product of the roots of a cubic equation using Vieta's formulas.",
        "Create a problem that requires solving an equation involving the sum of a geometric series. The common ratio should be negative and the sum should be finite.",
        "Generate a problem involving exponential growth. The problem should ask students to find the time at which a population exceeds a certain threshold, using a growth formula.",
        "Create a problem where students need to solve an inequality involving both absolute value and quadratic terms.",

        # Calculus
        "Create a problem that asks to find the derivative of a function involving both trigonometric and exponential terms. The function should be complex enough to require the product and chain rules.",
        "Generate a problem involving the calculation of a definite integral. The function should involve a rational expression that requires substitution to simplify the integral.",
        "Design a problem that involves using the Mean Value Theorem. The problem should ask to find the specific point where the derivative of the function equals the average rate of change over a given interval.",
        "Create a problem where students are required to determine the critical points and analyze the concavity of a given function using the first and second derivative tests.",
        "Generate a problem that involves evaluating an improper integral. The integral should be over an infinite interval or include a discontinuity within the domain of integration.",
        "Design a problem that requires finding the limit of a function as it approaches infinity. The problem should involve rational functions where students need to apply L'Hopital's Rule to evaluate the limit.",
        "Create a problem where students must solve a related rates problem, such as the rate of change of the distance between two moving objects. The problem should involve both linear and angular motion.",
        "Generate a problem involving the application of the Fundamental Theorem of Calculus. Students should be asked to compute the derivative of an integral that depends on a variable upper limit.",
        "Design a problem involving optimization, where students must find the maximum or minimum of a real-world function. The problem should require setting up an equation, finding critical points, and verifying the nature of the critical points.",
        "Create a question where students need to find the area between two curves. The curves should be non-linear and involve both polynomial and trigonometric terms.",
        "Design a problem that involves finding the derivative of an implicit function using implicit differentiation.",
        "Generate a problem where students are asked to solve a definite integral involving a piecewise function, with the intervals specified in the problem.",
        "Create a problem that requires the application of the Mean Value Theorem for integrals. Students should compute the average value of a function over a specified interval.",
        "Design a problem where students must determine the global maximum and minimum values of a function over a closed interval, and use the first and second derivative tests.",
        "Generate a problem where students need to find the antiderivative of a trigonometric function with a non-trivial coefficient, using substitution.",
        "Create a problem where students are asked to find the limit of a function that involves both exponential and logarithmic terms.",
        "Design a problem that requires finding the related rates of two objects moving in different directions. Students should calculate the rate at which the distance between the two objects changes.",
        "Generate a problem where students need to evaluate a limit using series expansion for an indeterminate form.",
        "Create a problem involving optimization, where students need to find the dimensions of a rectangle that maximize the area, given a fixed perimeter.",
        "Design a problem where students are asked to compute the area under a curve using numerical integration methods, such as the Trapezoidal Rule.",

        # Num Theory
        "Generate a number theory problem that involves finding the greatest common divisor (GCD) and least common multiple (LCM) of two large numbers. The numbers should be composite, and the problem should require applying the Euclidean algorithm.",
        "Create a problem where students must prove whether a given number is prime or composite using divisibility rules. The number should be large enough to require multiple steps for factorization.",
        "Design a problem that involves modular arithmetic. Students should be asked to solve a congruence equation, such as \\(x \\equiv 7 \\pmod{11}\\), and find all solutions within a given range.",
        "Generate a number theory problem that involves solving a Diophantine equation, such as finding integer solutions to a linear equation of the form \\(ax + by = c\\).",
        "Create a question that requires proving or disproving whether a number is a perfect square. The question should involve algebraic manipulation and prime factorization.",
        "Design a number theory problem involving the Chinese Remainder Theorem. The problem should ask for the solution to a system of simultaneous congruences with different moduli.",
        "Generate a problem that asks to determine the number of divisors of a large number. The question should require prime factorization and the formula for the number of divisors based on the exponents in the prime factorization.",
        "Create a problem that involves the use of the Fundamental Theorem of Arithmetic, asking to express a number as a product of prime factors and verify the uniqueness of the factorization.",
        "Design a problem involving Fermat's Little Theorem. Students should be asked to compute a large power of a number modulo a prime and use Fermat's Little Theorem to simplify the calculation.",
        "Generate a number theory problem that asks to find the sum of all integers less than or equal to a given number \\(n\\) that are coprime to \\(n\\). This should involve Euler's Totient Function and some basic properties of prime numbers.",
        "Create a problem involving the application of the Euclidean algorithm to find the greatest common divisor (GCD) of two large prime numbers.",
        "Design a problem where students need to prove whether a number is a perfect cube using modular arithmetic.",
        "Generate a number theory problem that asks students to find the modular inverse of a number in a given modulus.",
        "Create a question that requires solving a Diophantine equation with three variables, providing specific integer solutions.",
        "Design a problem where students need to determine if a given number is a Fibonacci number using divisibility rules and modular arithmetic.",
        "Generate a number theory problem that asks to find all solutions to a congruence equation involving both addition and multiplication modulo a prime.",
        "Create a problem that involves finding the least number that satisfies a system of modular congruences using the Chinese Remainder Theorem.",
        "Design a number theory problem involving the calculation of the Euler's Totient Function for a given large number.",
        "Generate a problem that asks students to prove that a number is a perfect power using its prime factorization.",
        "Create a problem involving the application of Fermat's Little Theorem to simplify calculations of large powers modulo a prime.",
        
        # Geometry
        "Generate a problem involving the properties of triangles. The question should ask to prove that a triangle is an equilateral, isosceles, or scalene based on given side lengths or angles.",
        "Create a question about the area and perimeter of a composite shape, such as a quadrilateral that includes a semicircle or a right triangle. Students should be asked to find both the area and the perimeter.",
        "Design a problem that involves calculating the volume and surface area of a cone. The cone should have a slant height provided, and students should be required to use both the surface area and volume formulas.",
        "Generate a problem about circles that involves the calculation of the length of an arc and the area of a sector. The problem should provide the central angle and radius of the circle.",
        "Create a geometric proof problem involving parallel lines and a transversal. Students should be asked to prove relationships such as corresponding angles, alternate interior angles, or consecutive interior angles.",
        "Design a problem involving the Pythagorean Theorem. The problem should ask to find the length of a side of a right triangle, given the lengths of the other two sides, and include a real-world context.",
        "Generate a problem involving coordinate geometry, where students must find the equation of a line given two points or calculate the distance between two points in the coordinate plane.",
        "Create a problem involving the volume and surface area of a cylinder. The cylinder should have both radius and height provided, and students should be required to calculate both the volume and the surface area.",
        "Design a problem involving the properties of polygons, where students must determine the number of diagonals in a polygon given the number of sides. Provide a formula and ask for a specific polygon.",
        "Generate a problem that involves angle relationships in polygons, specifically asking to find the sum of interior angles of a polygon with a given number of sides and use it to calculate individual angles in a regular polygon.",
        "Design a problem that involves finding the area of a sector of a circle, given the length of the arc and the central angle.",
        "Generate a problem involving the properties of a rhombus. Students should be asked to find the area and the lengths of the diagonals.",
        "Create a geometric proof problem that asks to prove that the opposite angles of two intersecting lines are congruent.",
        "Design a problem that involves calculating the volume of a cone when only the slant height and base radius are given.",
        "Generate a problem that involves finding the coordinates of the centroid of a triangle, given the vertices' coordinates.",
        "Create a problem where students are asked to calculate the area of a trapezoid given its height and bases, with one base being a function of the other.",
        "Design a problem that involves the properties of circles inscribed and circumscribed about a triangle. Students should be asked to find the radius of the incircle.",
        "Generate a problem that asks to prove that the sum of the interior angles of a quadrilateral is 360 degrees.",
        "Create a problem involving coordinate geometry, where students must determine the equation of the line that passes through two given points and is perpendicular to another given line.",
        "Design a problem that asks to find the surface area and volume of a sphere, given the radius, and requires applying the formulas for both.",

        # Probability and Statistics
        "Generate a probability problem involving the rolling of two fair dice. The question should ask for the probability of rolling a sum greater than 9, and require calculating the number of favorable outcomes.",
        "Create a problem where students need to compute the expected value of a random variable. The random variable should be related to a real-world situation, such as a game with different probabilities of winning and losing.",
        "Design a problem that involves conditional probability. Students should be asked to calculate the probability of an event given that another event has occurred, using Bayes' Theorem.",
        "Generate a problem that asks to find the variance and standard deviation of a discrete probability distribution. The distribution should have a range of values with non-equal probabilities.",
        "Create a problem involving the Central Limit Theorem. The question should ask for the sampling distribution of the sample mean of a population with a known mean and standard deviation, for a given sample size.",
        "Generate a question that requires calculating the confidence interval for a population mean. The sample size, sample mean, and sample standard deviation should be provided, and the students should be asked to interpret the results at a 95% confidence level.",
        "Design a problem where students must determine the probability of getting at least one heads in three flips of a biased coin. The bias should be specified (e.g., \\(P(\\text{Heads}) = 0.6\\)) and the problem should involve complementary probability.",
        "Create a problem involving hypothesis testing. The question should provide a dataset and ask students to test whether the sample mean is significantly different from a known population mean using a t-test.",
        "Design a problem that involves the use of the binomial distribution. Students should be asked to calculate the probability of getting exactly 4 successes in 10 trials, given a success probability of 0.3.",
        "Generate a statistics problem that involves analyzing a data set using regression analysis. The problem should ask students to find the line of best fit, interpret the slope and y-intercept, and assess the correlation between the variables.",
        "Generate a probability problem that involves a deck of cards. Students should be asked to calculate the probability of drawing exactly two aces when drawing four cards without replacement.",
        "Create a problem involving conditional probability with a real-world scenario, such as finding the probability of a customer purchasing a product given their age group.",
        "Design a problem that involves finding the expected value of a discrete random variable with multiple possible outcomes.",
        "Generate a problem that requires students to compute the standard deviation of a set of data points given in a table.",
        "Create a problem where students must determine the probability of getting exactly 5 heads in 8 flips of a biased coin, where the probability of heads is 0.4.",
        "Design a problem that involves hypothesis testing. Students should test whether a sample proportion is significantly different from a hypothesized value using a z-test.",
        "Generate a problem involving the application of the binomial distribution. Students should calculate the probability of getting at least 3 successes in 5 trials, with a probability of success of 0.6.",
        "Create a problem where students need to calculate the confidence interval for a population proportion, given sample size and observed successes.",
        "Design a problem that involves analyzing a scatter plot and using linear regression to determine the line of best fit.",
        "Generate a problem that requires applying the Central Limit Theorem to approximate the distribution of the sample mean for a large sample size.",
    ]}
    
    aug_dataset = Dataset.from_dict(aug_dict)
    aug_dataset = aug_dataset.map(function=make_map_fn('aug'), with_indices=True)
    aug_dataset.to_parquet(os.path.join(local_dir, 'aug.parquet'))