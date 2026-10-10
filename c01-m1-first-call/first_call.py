"""L0: compare LLM answers at different temperatures."""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

MODEL = "gpt-4.1-mini"
QUESTION = "Suggest one name for a new coffee shop. Reply with the name only."
TEMPERATURES = [0, 1.0, 1.5]
RUNS = 3


def ask(question: str, temperature: float) -> str:
    """Send one question to the model at the given temperature and return the reply text."""
    llm = ChatOpenAI(model=MODEL, temperature=temperature)
    response = llm.invoke(question)
    return response.content


def main() -> None:
    """Ask the same question RUNS times at each temperature and print the answers."""
    load_dotenv()
    for temp in TEMPERATURES:
        for run in range(1, RUNS + 1):
            answer = ask(QUESTION, temp)
            print(f"temp={temp} run={run}: {answer}")
        print("-" * 40)


if __name__ == "__main__":
    main()