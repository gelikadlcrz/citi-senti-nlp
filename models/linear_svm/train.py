"""Train CITI-Senti Linear Svm on the shared TF-IDF features.

Run from the citi-senti-nlp repository root:
    python -m models.linear_svm.train
"""
from shared.train_common import train


if __name__ == "__main__":
    train("linear_svm")
