# test_textpredict.py

import os
import shutil
import unittest

from datasets import load_dataset  # type: ignore

from textpredict import TextPredict


class TestTextPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tp = TextPredict()
        cls.dataset = load_dataset("imdb")
        cls.test_output_dir = "./test_output"
        if not os.path.exists(cls.test_output_dir):
            os.makedirs(cls.test_output_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)

    def test_analyse_sentiment(self):
        result = self.tp.analyse("I love using this package!", task="sentiment")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print("Sentiment Analysis Result:", result)

    def test_analyse_emotion(self):
        result = self.tp.analyse("I am excited about this!", task="emotion")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print("Emotion Analysis Result:", result)

    def test_analyse_zeroshot(self):
        result = self.tp.analyse(
            "This package is great for zero-shot learning.",
            task="zeroshot",
            class_list=["positive", "negative", "neutral"],
        )
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print("Zero-Shot Classification Result:", result)

    def test_fine_tune_model(self):
        self.tp.tune_model(
            task="sentiment",
            training_data=self.dataset["train"],
            eval_data=self.dataset["test"],
            output_dir=os.path.join(self.test_output_dir, "fine_tuned_sentiment_model"),
            num_train_epochs=1,
            batch_size=8,
            learning_rate=2e-5,
            early_stopping_patience=3,
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_output_dir, "fine_tuned_sentiment_model")
            )
        )
        print("Model fine-tuned and saved successfully.")

    def test_evaluate_model(self):
        metrics = self.tp.evaluate_model(
            task="sentiment", eval_data=self.dataset["test"]
        )
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        print("Evaluation metrics:", metrics)

    def test_save_model(self):
        self.tp.save_model(
            task="sentiment",
            output_dir=os.path.join(self.test_output_dir, "saved_sentiment_model"),
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_output_dir, "saved_sentiment_model"))
        )
        print("Model saved successfully.")

    def test_load_model(self):
        self.tp.load_model(
            task="sentiment",
            model_dir=os.path.join(self.test_output_dir, "saved_sentiment_model"),
        )
        result = self.tp.analyse(
            "I love using this package after fine-tuning!", task="sentiment"
        )
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print("Model loaded and used for analysis successfully.")


if __name__ == "__main__":
    unittest.main()
