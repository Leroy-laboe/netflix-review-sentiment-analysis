# Netflix Review Sentiment Analysis

**A Natural Language Processing project that classifies Netflix Google Play reviews as Positive, Neutral, or Negative and compares multiple text-representation and classification approaches.**

Built as a group NLP project using **Python, NLTK, scikit-learn, TF-IDF, Word2Vec, SMOTE, and Streamlit**.

---

## Overview

This project analyses user-written Netflix reviews from the Google Play Store and predicts one of three sentiment classes:

- **Positive** — rating of 4 or 5 stars
- **Neutral** — rating of 3 stars
- **Negative** — rating of 1 or 2 stars

The work covers the full NLP workflow:

```text
Raw Netflix Reviews
        │
        ▼
Data Cleaning
        │
        ▼
English-Language Filtering
        │
        ▼
Text Preprocessing
        │
        ├── lowercase
        ├── punctuation removal
        ├── stop-word removal
        └── lemmatization
        │
        ▼
Feature Engineering
        │
        ├── TF-IDF
        ├── CountVectorizer experiments
        ├── thumbs-up metadata
        └── Word2Vec embeddings
        │
        ▼
Model Training & Comparison
        │
        ├── Logistic Regression
        ├── Linear SVM
        └── Word2Vec + Logistic Regression
        │
        ▼
3-Class Sentiment Prediction
```

A Streamlit application is included for entering a review and running sentiment inference with the saved model artifacts.

---

## Dataset

The repository includes `netflix_reviews.csv`, containing Netflix Google Play review data.

The notebook records:

| Stage | Reviews |
| --- | ---: |
| Raw dataset | **57,486** |
| After null / duplicate / short-review cleaning | **50,064** |
| After English-language filtering | **44,626** |

The analysis keeps the following core fields:

- review text (`content`)
- star rating (`score`)
- helpful-vote count (`thumbsUpCount`)

> Sentiment labels are derived from the review's star rating, while the text itself is used for NLP modelling.

---

## Text Preprocessing

The preprocessing pipeline uses NLTK and regular expressions to:

1. convert text to lowercase;
2. remove non-alphabetic characters;
3. tokenize the review;
4. remove English stop words;
5. lemmatize remaining words.

The notebook also performs exploratory analysis using:

- sentiment-distribution plots;
- word clouds;
- bigram-frequency analysis;
- rating vs. thumbs-up visualisation.

---

## Feature Engineering

### TF-IDF

The main sparse-text experiments use:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

The notebook also compares alternate vocabulary sizes and CountVectorizer features.

### Review metadata

The `thumbsUpCount` feature is scaled and combined with the TF-IDF feature matrix for the traditional-model experiments.

### Class balancing

The training split is balanced using **SMOTE** before training the Logistic Regression and Linear SVM models.

### Word2Vec

A separate semantic representation is created using a **100-dimensional Skip-gram Word2Vec model**.

Each review is represented by the mean vector of the words available in the trained vocabulary.

---

## Model Comparison

The project notebook records the following held-out test results:

| Model | Recorded Accuracy |
| --- | ---: |
| **Word2Vec + Logistic Regression** | **78.15%** |
| TF-IDF + Logistic Regression | 71.81% |
| TF-IDF + Linear SVM | 70.28% |

The models were evaluated on a stratified 80/20 train-test split.

The notebook also includes:

- precision, recall and F1-score reports;
- confusion matrices;
- model-accuracy comparisons.

### Important class-level observation

The three-class problem is imbalanced and the **Neutral** class is substantially harder to classify than Positive or Negative reviews.

That limitation is visible in the class-level evaluation metrics and is useful context when interpreting overall accuracy.

---

## Streamlit Application

The repository includes a simple Streamlit interface for sentiment prediction.

The app:

1. accepts a Netflix review;
2. applies the same cleaning / lemmatization steps;
3. transforms the text with the saved vectorizer;
4. runs the saved classifier;
5. displays **Positive**, **Neutral**, or **Negative**.

Run it with:

```bash
streamlit run app.py
```

---

## Tech Stack

| Area | Technologies |
| --- | --- |
| Language | Python |
| NLP | NLTK |
| Traditional ML | scikit-learn |
| Text Features | TF-IDF, CountVectorizer |
| Embeddings | Gensim Word2Vec |
| Class Balancing | SMOTE / imbalanced-learn |
| Data Analysis | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| App | Streamlit |
| Model Persistence | joblib |

---

## Repository Structure

```text
.
├── notebooks/
│   └── netflix_sentiment_analysis.ipynb
├── app.py
├── model.pkl
├── vectorizer.pkl
├── netflix_reviews.csv
├── requirements.txt
├── requirements-notebook.txt
└── README.md
```

---

## Run the App

### 1. Clone

```bash
git clone https://github.com/Leroy-laboe/Sentiment-Analysis-of-Netflix-Play-Store-Reviews-.git
cd Sentiment-Analysis-of-Netflix-Play-Store-Reviews-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install application dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

---

## Reproduce the Notebook

For the full analysis environment:

```bash
pip install -r requirements-notebook.txt
```

Then open:

```text
notebooks/netflix_sentiment_analysis.ipynb
```

The dataset is expected at the repository root as:

```text
netflix_reviews.csv
```

---

## Project Team

This was a **group NLP project** developed by:

- **Leroy Nyasha Mangwarara**
- **Tadiwanashe**
- **Fathima Zuha**

The repository is maintained here as part of Leroy's technical portfolio, while preserving the group-project attribution.

---

## What This Project Demonstrates

- practical text cleaning and preprocessing;
- multi-class sentiment classification;
- feature-engineering experiments;
- handling class imbalance;
- comparing sparse and semantic text representations;
- evaluating models beyond headline accuracy;
- packaging a trained NLP classifier into a small web application.

---

## Author / Maintainer

**Leroy Nyasha Mangwarara**

Computer Science · Data Science · Software Engineering · Applied AI

[GitHub](https://github.com/Leroy-laboe) · [LinkedIn](https://www.linkedin.com/in/leroy-nyasha-mangwarara-86185a302/) · [Email](mailto:mangwararaleroy@gmail.com)
