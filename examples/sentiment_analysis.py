# examples/sentiment_analysis.py

from textpredict import TextPredict


def main():
    # Initialize the TextPredict class
    tp = TextPredict()

    # Example texts for sentiment analysis
    texts = [
        "I love using this package!",
        "This is the worst experience I've ever had.",
        "I feel great about this product.",
        "I am not happy with the service.",
        "This is a neutral statement.",
    ]

    # Perform sentiment analysis on each text
    for text in texts:
        sentiment_result = tp.analyse(text, task="sentiment")
        print(f"Text: {text}")
        print("Sentiment Analysis Result:", sentiment_result)
        print()


if __name__ == "__main__":
    main()
