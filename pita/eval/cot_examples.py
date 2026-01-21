from __future__ import annotations

# Standard 8-shot examples for GSM8K from the original Chain-of-Thought paper (Wei et al. 2022)
GSM8K_8SHOT = [
    {
        "q": "There are 15 balls in a basket. 3 of them are red and the rest are blue. How many blue balls are there?",
        "a": "There are 15 balls in total. 3 are red. So there are 15 - 3 = 12 blue balls. The answer is \\boxed{12}.",
    },
    {
        "q": "If there are 3 cars in the parking lot and 2 more arrive, how many cars are in the parking lot?",
        "a": "There are 3 cars originally. 2 more arrive. 3 + 2 = 5. The answer is \\boxed{5}.",
    },
    {
        "q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "a": "Olivia had $23. 5 bagels for $3 each costs 5 * 3 = $15. 23 - 15 = 8. The answer is \\boxed{8}.",
    },
    {
        "q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "a": "Michael started with 58 golf balls. He lost 23, so he has 58 - 23 = 35. Then he lost 2 more, so 35 - 2 = 33. The answer is \\boxed{33}.",
    },
    {
        "q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "a": "Leah had 32 and her sister had 42. That means there were 32 + 42 = 74 chocolates. 35 were eaten. So 74 - 35 = 39. The answer is \\boxed{39}.",
    },
    {
        "q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "a": "Jason started with 20 lollipops. He now has 12, so he gave 20 - 12 = 8 lollipops to Denny. The answer is \\boxed{8}.",
    },
    {
        "q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "a": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, that is 2 + 2 = 4 more toys. 5 + 4 = 9. The answer is \\boxed{9}.",
    },
    {
        "q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "a": "There were 9 computers originally. From monday to thursday is 4 days. 5 computers were installed each day, so 4 * 5 = 20 computers were added. 9 + 20 = 29. The answer is \\boxed{29}.",
    },
]

# Standard 8-shot examples for MATH
MATH_8SHOT = [
    {
        "q": "What is the degree of the polynomial $P(x) = (x^3 + 1)^2(x^2 - 1)^3$?",
        "a": "The degree of $(x^3+1)^2$ is $3 \\cdot 2 = 6$. The degree of $(x^2-1)^3$ is $2 \\cdot 3 = 6$. When we multiply these two polynomials, their degrees add: $6+6=12$. The answer is \\boxed{12}.",
    },
    {
        "q": "Find the value of $x$ such that $2^{x+3} = 32$.",
        "a": "We know $32 = 2^5$. So $2^{x+3} = 2^5$. This implies $x+3 = 5$, which means $x = 2$. The answer is \\boxed{2}.",
    },
    {
        "q": "If $f(x) = x^2 - 3x + 2$, find $f(f(1))$.",
        "a": "First, find $f(1)$: $1^2 - 3(1) + 2 = 1 - 3 + 2 = 0$. Now find $f(0)$: $0^2 - 3(0) + 2 = 2$. The answer is \\boxed{2}.",
    },
    {
        "q": "The sum of two numbers is 10 and their product is 21. What is the sum of their squares?",
        "a": "Let the numbers be $x$ and $y$. $x+y=10$ and $xy=21$. We want $x^2+y^2$. We know $(x+y)^2 = x^2+y^2+2xy$. So $10^2 = x^2+y^2+2(21)$, which is $100 = x^2+y^2+42$. Thus $x^2+y^2=58$. The answer is \\boxed{58}.",
    },
]

# Standard 8-shot examples for AIME
AIME_8SHOT = [
    {
        "q": "Find the number of ordered triples $(a,b,c)$ of positive integers such that $abc=108$.",
        "a": "The prime factorization of $108$ is $2^2 \cdot 3^3$. Let $a = 2^{a_1}3^{a_2}$, $b = 2^{b_1}3^{b_2}$, and $c = 2^{c_1}3^{c_2}$. Then $a_1+b_1+c_1=2$ and $a_2+b_2+c_2=3$. The number of solutions to $a_1+b_1+c_1=2$ is $\binom{2+3-1}{3-1}=6$. For $a_2+b_2+c_2=3$, it is $\binom{3+3-1}{3-1}=10$. Total triples: $6 \cdot 10 = 60$. The answer is \\boxed{60}.",
    },
    {
        "q": "A function $f$ is defined on the complex numbers by $f(z) = (z-1)^2$. How many solutions does the equation $f(f(z)) = 0$ have?",
        "a": "We have $f(f(z)) = ((z-1)^2 - 1)^2 = 0$. This means $(z-1)^2 - 1 = 0$, so $(z-1)^2 = 1$. Thus $z-1 = 1$ or $z-1 = -1$, giving $z=2$ or $z=0$. Both are double roots of the degree-4 polynomial. Distinct solutions are 0 and 2. The answer is \\boxed{2}.",
    },
    {
        "q": "Find the number of positive integers $n < 100$ such that $n$ is a multiple of 3 and $n+1$ is a multiple of 4.",
        "a": "We have $n = 3k$ and $n = 4m - 1$. So $3k = 4m - 1$, or $4m - 3k = 1$. A solution is $m=1, k=1$, giving $n=3$. The general solution is $n = 3 + 12j$. For $n < 100$, $12j < 97$, so $j \in \{0, 1, ..., 8\}$. There are 9 such integers. The answer is \\boxed{9}.",
    },
]


def get_8shot_prompt(dataset_name: str, question: str) -> str:
    if dataset_name == "GSM8K":
        examples = GSM8K_8SHOT
    elif dataset_name == "MATH":
        examples = MATH_8SHOT
    elif dataset_name == "AIME":
        examples = AIME_8SHOT
    else:
        # Fallback to direct prompt if no examples defined
        return (
            f"Problem:\n\n{question} Write your answer inside \\boxed{{}}.\n\nSolution:"
        )

    prompt = ""
    for ex in examples:
        prompt += f"Problem:\n{ex['q']}\n\nSolution: {ex['a']}\n\n"

    prompt += f"Problem:\n{question}\n\nSolution: Let's think step by step."
    return prompt
