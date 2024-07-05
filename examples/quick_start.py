# examples/quick_start.py

from textpredict import TextPredict


def main():
    # Initialize the TextPredict class
    tp = TextPredict()

    # Perform sentiment analysis
    sentiment_result = tp.analyse(
        "I am very happy with this package!", task="sentiment"
    )
    print("Sentiment Analysis Result:", sentiment_result)

    # Perform emotion analysis
    emotion_result = tp.analyse("I am very happy with this package!", task="emotion")
    print("Emotion Analysis Result:", emotion_result)

    # Perform zero-shot classification
    zeroshot_result = tp.analyse(
        "I am very happy with this package!",
        task="zeroshot",
        class_list=["positive", "negative", "neutral"],
    )
    print("Zero-Shot Analysis Result:", zeroshot_result)


if __name__ == "__main__":
    main()
