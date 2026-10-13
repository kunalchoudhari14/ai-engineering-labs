"""L2: LCEL chains with the pipe operator, RunnableLambda and RunnableParallel."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

MODEL = "gpt-4.1-mini"


def format_prompt(inputs: dict) -> str:
    """Build the joke prompt string from the input dict."""
    return f"Tell me short joke about {inputs['topic']}."


def main() -> None:
    """Run each LCEL chain demo and print its output."""

    load_dotenv()
    llm = ChatOpenAI(model=MODEL, temperature=0)
    joke_prompt = PromptTemplate.from_template("Tell me short joke about {topic}.")
    joke_chain = joke_prompt | llm | StrOutputParser()
    joke_text = joke_chain.invoke({"topic": "data analysts"})
    print(joke_text)

    joke_chain_no_parser = joke_prompt | llm
    joke_message = joke_chain_no_parser.invoke({"topic": "cats"})
    print(type(joke_message))

    lambda_chain = RunnableLambda(format_prompt) | llm | StrOutputParser()
    sql_joke = lambda_chain.invoke({"topic": "SQL"})
    print(sql_joke)

    summary_prompt = PromptTemplate.from_template(
        "Summarize this in one sentence: {text}"
    )

    sentiment_prompt = PromptTemplate.from_template(
        "Classify the sentiment of this text as positive, negative or neutral. Reply with one word: {text}"
    )
    translation_prompt = PromptTemplate.from_template(
        "Translate this text into French: {text}"
    )
    review_text = "The dashboard loads fast and the charts are clear, but exporting to Excel keeps failing."

    summary_chain = summary_prompt | llm | StrOutputParser()
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    translation_chain = translation_prompt | llm | StrOutputParser()

    review_analysis = RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
        translation=translation_chain,
    )

    review_results = review_analysis.invoke({"text": review_text})
    print(review_results)
    print(type(review_results))


    joke_translation_chain = {"text": joke_chain} | translation_chain
    translated_joke = joke_translation_chain.invoke({"topic": "Python"})
    print(translated_joke)

if __name__ == "__main__":
    main()
