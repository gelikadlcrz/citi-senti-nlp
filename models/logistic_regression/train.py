"""Train CITI-Senti Logistic Regression on the shared TF-IDF features.

Run from the citi-senti-nlp repository root:
    python -m models.logistic_regression.train
"""
from shared.train_common import train


if __name__ == "__main__":
    train("logreg")
