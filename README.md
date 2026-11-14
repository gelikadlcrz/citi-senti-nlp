# CITI-Senti

**Multilingual Citizen Feedback Sentiment Analysis**

CITI-Senti is a Natural Language Processing (NLP) research project focused on trilingual aspect-based sentiment analysis (ABSA) of citizen feedback concerning Baguio City government policies, programs, ordinances, resolutions, and public issues. The study investigates multilingual aspect detection and aspect-level sentiment classification using Glot500-m, with Logistic Regression and Linear SVM serving as traditional machine learning baselines.

The study aims to evaluate the effectiveness of **Glot500-m** for sentiment classification by comparing its performance against two traditional machine learning baselines: Logistic Regression and Linear Support Vector Machine (SVM).

This repository contains the source code, experiments, and evaluation components for CITI-Senti's NLP models, as well as the planned inference pipeline and API integration.

## Models

CITI-Senti investigates three machine learning approaches for trilingual aspect-based sentiment analysis (ABSA):

| Model | Approach | Role |
|---|---|---|
| Logistic Regression | Traditional Machine Learning | Baseline 1 |
| Linear SVM | Traditional Machine Learning | Baseline 2 |
| Glot500-m | Multilingual Transformer | Proposed Model |

Each approach is developed to perform two primary NLP tasks:

1. **Multi-label Aspect Detection** — Identifies one or more aspects discussed in citizen feedback based on the predefined aspect taxonomy.
2. **Aspect-Based Sentiment Classification** — Determines the sentiment (Positive, Negative, or Neutral) expressed toward each identified aspect.

The traditional machine learning baselines use shared Term Frequency–Inverse Document Frequency (TF-IDF) feature representations to ensure consistency in feature extraction. Glot500-m uses pretrained multilingual representations and will be fine-tuned for both classification tasks.

All three approaches will be evaluated using the same training, validation, and testing partitions and standardized evaluation metrics to support a comparative analysis of their performance across English, Tagalog, Ilokano, and code-switched citizen feedback.
## Repository Structure

```text
citi-senti-nlp/
├── data/                # Dataset information and partitions
├── models/
│   ├── logistic_regression/
│   ├── linear_svm/
│   └── glot500m/
├── notebooks/           # Corpus analysis and experiments
├── evaluation/          # Model evaluation and comparison
├── inference/           # Sentiment prediction pipeline
├── api/                 # API integration
├── results/             # Evaluation results and visualizations
├── tests/               # Testing and validation
├── docs/                # Documentation
└── shared/              # Shared utilities
```

## Evaluation

Model performance will be assessed using standard classification metrics, including accuracy, precision, recall, and F1-score. Confusion matrices will also be used to examine classification performance across sentiment categories.

The evaluation aims to identify the strengths and limitations of each approach when applied to multilingual citizen feedback.

## Application Integration

The selected sentiment classification model is intended to be integrated into the CITI-Senti web application through a Python-based inference pipeline and REST API using FastAPI.

This integration will allow the application to process submitted citizen feedback and return sentiment predictions.

## Project Development

CITI-Senti is currently under active development. Model implementations, experimental results, and integration components will be added as development progresses.

Setup instructions and model execution procedures will be documented as the implementations become available.

## Data Availability

The project uses an annotated corpus of citizen feedback. Dataset files are not included in this public repository by default to protect potentially sensitive information and comply with applicable data-sharing restrictions.

[Shared TF-IDF Documentation](docs/TFIDF_SETUP.md)