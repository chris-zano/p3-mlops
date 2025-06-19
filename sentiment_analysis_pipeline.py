"""
module to perform sentiment analysis on using Hugging Face transformers
"""
# pylint: disable=import-error
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def main():
    """Main function to run sentiment analysis on a text file"""

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    with open("generated-text.txt", "r", encoding="utf-8") as file:
        contents = file.read()
        results = classifier(contents)



    for result in results:
        sentiment = LABEL_MAP.get(result["label"], "Unknown")
        print(f"Sentiment: {sentiment}, Confidence: {result['score']:.2%}")

if __name__ == "__main__":
    main()
