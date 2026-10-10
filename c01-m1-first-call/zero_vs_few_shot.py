"""L0: compare zero-shot and few-shot prompts on the same review."""

from dotenv import load_dotenv
from first_call import ask

REVIEW = "Delivery was 5 days late, but the product itself is excellent."

ZERO_SHOT = f"Classify the sentiment of this review: {REVIEW}"

FEW_SHOT = f"""Classify the sentiment of each review as positive, negative or mixed.

Review: The app crashes every time I open it.
Sentiment: negative

Review: <write a clearly positive review>
Sentiment: positive

Review: <write a review with one good and one bad point>
Sentiment: mixed

Review: {REVIEW}
Sentiment:"""


def main() -> None:
    """Run the same review through a zero-shot and a few-shot prompt and print both answers."""
    load_dotenv()
    print("ZERO-SHOT:", ask(ZERO_SHOT, 0))
    print("-" * 40)
    print("FEW-SHOT:", ask(FEW_SHOT, 0))


if __name__ == "__main__":
    main()
