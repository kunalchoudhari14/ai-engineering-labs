"""L1: compare direct, chain-of-thought and self-consistency prompts."""

from collections import Counter

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

MODEL = "gpt-4.1-mini"
RUNS = 5

APPLES = (
    "A cafeteria had 23 apples. They used 20 for lunch and bought 6 more. "
    "How many apples do they have now?"
)

SISTER = (
    "When I was 6, my sister was half my age. Now I am 70. "
    "How old is my sister?"
)

TEMPLATES = {
    "DIRECT": PromptTemplate.from_template(
        "{question}\nAnswer with the number only."
    ),
    "COT": PromptTemplate.from_template(
        "{question}\nLet's think step by step, then give the final answer."
    ),
}

VOTE = PromptTemplate.from_template(
    "{question}\nLet's think step by step. "
    "End with a last line exactly like: Final answer: <number>"
)


def run(llm: ChatOpenAI, template: PromptTemplate, question: str) -> str:
    """Fill the template with the question, send it to the model and return the reply text."""
    prompt = template.format(question=question)
    return llm.invoke(prompt).content


def final_answer(reply: str) -> str:
    """Return the text after 'Final answer:' on the reply's last line."""
    return reply.strip().splitlines()[-1].split("Final answer:")[-1].strip()


def self_consistency(question: str) -> str:
    """Ask the CoT question RUNS times at temperature > 0 and return the majority answer."""
    llm = ChatOpenAI(model=MODEL, temperature=1.0)
    answers = [final_answer(run(llm, VOTE, question)) for _ in range(RUNS)]
    print("All answers:", answers)
    return Counter(answers).most_common(1)[0][0]


def main() -> None:
    """Run the same question through each technique and print the answers."""
    load_dotenv()
    llm = ChatOpenAI(model=MODEL, temperature=0)
    for name, template in TEMPLATES.items():
        print(f"{name}:\n{run(llm, template, APPLES)}")
        print("-" * 40)
    print("SELF-CONSISTENCY:", self_consistency(SISTER))


if __name__ == "__main__":
    main()
