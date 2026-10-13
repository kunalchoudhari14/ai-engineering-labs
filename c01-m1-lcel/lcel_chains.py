"""L2: LCEL chains with the pipe operator, RunnableLambda and RunnableParallel."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

MODEL = "gpt-4.1-mini"
JOKE_PROMPT = "Tell me a short joke about {topic}."
SUMMARY_PROMPT = "Summarize this in one sentence: {text}"
SENTIMENT_PROMPT = "Classify the sentiment of this text as positive, negative or neutral. Reply with one word: {text}"
TRANSLATION_PROMPT = "Translate this text into French: {text}"
REVIEW_TEXT = "The dashboard loads fast and the charts are clear, but exporting to Excel keeps failing."


def format_prompt(inputs: dict) -> str:
    """Build the joke prompt string from the input dict."""
    return JOKE_PROMPT.format(topic=inputs["topic"])


def make_text_chain(llm: ChatOpenAI, template: str) -> Runnable:
    """Build a prompt | llm | StrOutputParser chain from a template string."""
    return PromptTemplate.from_template(template) | llm | StrOutputParser()


def demo_parser(llm: ChatOpenAI) -> None:
    """Show the output type with and without StrOutputParser."""
    joke_prompt = PromptTemplate.from_template(JOKE_PROMPT)
    joke_chain = joke_prompt | llm | StrOutputParser()
    joke_text = joke_chain.invoke({"topic": "data analysts"})
    print(joke_text)

    joke_chain_no_parser = joke_prompt | llm
    joke_message = joke_chain_no_parser.invoke({"topic": "cats"})
    print(type(joke_message))


def demo_lambda(llm: ChatOpenAI) -> None:
    """Use a plain Python function (via RunnableLambda) as the first chain step."""
    lambda_chain = RunnableLambda(format_prompt) | llm | StrOutputParser()
    sql_joke = lambda_chain.invoke({"topic": "SQL"})
    print(sql_joke)


def demo_parallel(llm: ChatOpenAI) -> None:
    """Run summary, sentiment and translation on one text at the same time."""
    summary_chain = make_text_chain(llm, SUMMARY_PROMPT)
    sentiment_chain = make_text_chain(llm, SENTIMENT_PROMPT)
    translation_chain = make_text_chain(llm, TRANSLATION_PROMPT)
    review_analysis = RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
        translation=translation_chain,
    )

    review_results = review_analysis.invoke({"text": REVIEW_TEXT})
    print(review_results)
    print(type(review_results))


def demo_sequence(llm: ChatOpenAI) -> None:
    """Chain two chains: write a joke, then translate it."""
    joke_chain = make_text_chain(llm, JOKE_PROMPT)
    translation_chain = make_text_chain(llm, TRANSLATION_PROMPT)
    joke_translation_chain = {"text": joke_chain} | translation_chain
    translated_joke = joke_translation_chain.invoke({"topic": "Python"})
    print(translated_joke)


def main() -> None:
    """Run each LCEL chain demo and print its output."""
    load_dotenv()
    llm = ChatOpenAI(model=MODEL, temperature=0)
    demo_parser(llm)
    demo_lambda(llm)
    demo_parallel(llm)
    demo_sequence(llm)


if __name__ == "__main__":
    main()
