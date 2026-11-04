# CITI-Senti

**Multilingual Citizen Feedback Sentiment Analysis**

CITI-Senti is a Natural Language Processing (NLP) research project focused on sentiment classification of multilingual citizen feedback. The project investigates the use of traditional machine learning and multilingual transformer-based approaches to identify sentiments expressed in citizen feedback.

The study aims to evaluate the effectiveness of **Glot500-m** for sentiment classification by comparing its performance against two traditional machine learning baselines: Logistic Regression and Linear Support Vector Machine (SVM).

This repository contains the source code, experiments, and evaluation components for CITI-Senti's NLP models, as well as the planned inference pipeline and API integration.

## Models

The project explores three sentiment classification models:

| Model | Approach | Role |
|---|---|---|
| Logistic Regression | Traditional Machine Learning | Baseline |
| Linear SVM | Traditional Machine Learning | Baseline |
| Glot500-m | Multilingual Transformer | Proposed Model |

The traditional machine learning models are planned to use TF-IDF feature representations, while Glot500-m will be fine-tuned on the project's annotated citizen feedback corpus.

All models will be evaluated using consistent dataset partitions and classification metrics to support a comparative analysis.

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