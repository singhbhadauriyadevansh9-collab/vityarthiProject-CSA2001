# Project Statement
## Fake News Detection using Machine Learning

> **Domain:** Natural Language Processing / Machine Learning
> **Language:** Python 3.7+
> **Type:** Supervised Binary Text Classification

---

## Project Details

| Field               | Details                                              |
|---------------------|------------------------------------------------------|
| Project Title       | Fake News Detection using Machine Learning           |
| Domain              | Natural Language Processing / Machine Learning       |
| Programming Language| Python 3.7+                                          |
| Key Libraries       | scikit-learn, NLTK, pandas, matplotlib, seaborn      |
| Dataset             | fake_news_dataset.csv (100 labeled articles)         |
| Project Type        | Supervised Binary Text Classification                |
| Difficulty Level    | Intermediate                                         |

---

## 1. Introduction

The rapid spread of misinformation and fabricated news stories on digital platforms has become one of the most pressing challenges of the modern information age. Fake news misleads the public, influences elections, damages reputations, and erodes trust in legitimate media institutions. Manual fact-checking is time-consuming and cannot scale to the volume of content published online every day.

This project addresses the problem by building an automated **Fake News Detection** system powered by Machine Learning and Natural Language Processing (NLP). The system analyzes the textual content of a news article and classifies it as either **REAL** or **FAKE** with measurable accuracy.

---

## 2. Problem Statement

Given a news article consisting of a headline and body text, the objective is to build a supervised machine learning model that can automatically classify the article as:

- **REAL** — A factually accurate, credible news article
- **FAKE** — A fabricated, misleading, or satirical article

The system must preprocess raw text, extract meaningful numerical features, train a classifier, evaluate its performance on unseen data, and support predictions on new articles.

---

## 3. Objectives

- Collect and prepare a labeled dataset of real and fake news articles
- Apply NLP preprocessing: tokenization, stopword removal, and stemming
- Convert text data into numerical features using TF-IDF vectorization
- Train and compare multiple machine learning classifiers
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix
- Identify the best-performing model and save it for deployment
- Enable prediction on new, unseen news articles

---

## 4. Dataset Description

The dataset `fake_news_dataset.csv` is a manually curated collection of **100 labeled news articles** covering topics such as science, health, politics, space, and global events.

| Column  | Type    | Description                            |
|---------|---------|----------------------------------------|
| `id`    | Integer | Unique article identifier              |
| `title` | String  | Headline / title of the article        |
| `text`  | String  | Full body text of the article          |
| `label` | String  | Ground truth label: `REAL` or `FAKE`   |

**Class Distribution:** 50 REAL articles | 50 FAKE articles *(Balanced dataset)*

---

## 5. Methodology

### 5.1 Text Preprocessing

Each article's title and body text are combined and passed through the following NLP pipeline:

1. Lowercase conversion of all characters
2. Removal of URLs, hyperlinks, and web addresses
3. Removal of numeric digits
4. Removal of punctuation marks
5. Removal of English stopwords using NLTK corpus
6. Porter Stemming to reduce words to their root form

### 5.2 Feature Extraction — TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** converts cleaned text into numerical feature vectors. It assigns higher weights to words that are frequent in a document but rare across the corpus, making it ideal for distinguishing REAL from FAKE content.

| Parameter      | Value  | Purpose                                        |
|----------------|--------|------------------------------------------------|
| `max_features` | 5,000  | Top 5000 words by corpus frequency             |
| `ngram_range`  | (1, 2) | Include unigrams and bigrams                   |
| `min_df`       | 2      | Ignore words appearing in fewer than 2 docs    |
| `sublinear_tf` | True   | Apply log normalization to term frequencies    |

### 5.3 Train / Test Split

The dataset is divided into **80% training** and **20% testing** sets using stratified sampling to preserve class balance.

### 5.4 Model Training

Four machine learning classifiers are trained and compared:

| Model                          | Why Used                                                                        |
|-------------------------------|---------------------------------------------------------------------------------|
| Logistic Regression            | Efficient linear classifier; strong baseline for high-dimensional sparse data   |
| Passive Aggressive Classifier  | Online learning; updates only on misclassified samples; fast for text tasks     |
| Naive Bayes (Multinomial)      | Probabilistic; assumes feature independence; fast and well-suited to text       |
| Random Forest                  | Ensemble of 100 decision trees; robust; handles non-linear patterns             |

---

## 6. Evaluation Metrics

Model performance is assessed using the following standard classification metrics:

| Metric           | Description                                                   |
|------------------|---------------------------------------------------------------|
| Accuracy         | Percentage of correctly classified articles                   |
| Precision        | Of all predicted FAKE articles, how many are truly FAKE       |
| Recall           | Of all actual FAKE articles, how many were correctly found    |
| F1-Score         | Harmonic mean of Precision and Recall                         |
| Confusion Matrix | 2x2 grid showing True Positives, True Negatives, FP, FN      |

The model with the highest accuracy is automatically selected as the **best model** and saved to `fake_news_model.pkl`.

---

## 7. Project Files

```
fake-news-detection/
│
├── fake_news_detection.py     # Main ML pipeline script
├── fake_news_dataset.csv      # Labeled dataset (100 articles)
├── fake_news_model.pkl        # Saved trained model (generated on run)
├── fake_news_analysis.png     # Visualization charts (generated on run)
├── README.md                  # Full project documentation
└── statement.md               # This project statement
```

| File                      | Description                                              |
|---------------------------|----------------------------------------------------------|
| `fake_news_detection.py`  | Main Python script — full ML pipeline                    |
| `fake_news_dataset.csv`   | 100 labeled REAL and FAKE news articles                  |
| `fake_news_model.pkl`     | Serialized trained model + TF-IDF vectorizer             |
| `fake_news_analysis.png`  | 4-panel visualization chart                              |
| `README.md`               | Full project documentation and usage guide               |
| `statement.md`            | This project statement document                          |

---

## 8. How to Run the Project

### Step 1 — Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

### Step 2 — Place Files in the Same Folder

Make sure the following two files are in the same directory:

```
fake_news_detection.py
fake_news_dataset.csv
```

### Step 3 — Run the Script

```bash
python fake_news_detection.py
```

### Step 4 — Review Outputs

- Accuracy and classification report printed in the terminal
- `fake_news_analysis.png` saved with 4 visualization charts
- `fake_news_model.pkl` saved with the best trained model

---

## 9. ML Pipeline Flow

```
Raw Text (title + body)
        |
        v
 [Text Preprocessing]
  - Lowercase
  - Remove URLs / numbers / punctuation
  - Remove stopwords (NLTK)
  - Porter Stemming
        |
        v
 [TF-IDF Vectorization]
  - max_features = 5000
  - ngram_range  = (1, 2)
  - sublinear_tf = True
        |
        v
 [Model Training]
  - Logistic Regression
  - Passive Aggressive Classifier
  - Naive Bayes (Multinomial)
  - Random Forest
        |
        v
 [Evaluation]
  - Accuracy, Precision, Recall, F1
  - Confusion Matrix
        |
        v
 [Best Model Saved]
  - fake_news_model.pkl
```

---

## 10. Expected Outcomes

- A trained ML model capable of classifying news articles as REAL or FAKE
- Model accuracy report with precision, recall, and F1-score per class
- Visual analysis: dataset distribution, model comparison, and confusion matrix
- Saved model artifact ready for integration into a web application or API
- Reusable prediction function for classifying any new news article

---

## 11. Sample Prediction Code

```python
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [stemmer.stem(w) for w in text.split()
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# Load saved model
with open('fake_news_model.pkl', 'rb') as f:
    saved = pickle.load(f)

model      = saved['model']
vectorizer = saved['vectorizer']

# Predict
def predict(text):
    cleaned = preprocess_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    return 'REAL' if pred == 1 else 'FAKE'

# Example usage
article = "Scientists confirm new vaccine achieves 95% efficacy in clinical trials."
print(predict(article))  # Output: REAL
```

---

## 12. Future Scope

- Extend with a larger real-world dataset such as LIAR or FakeNewsNet
- Implement deep learning models: LSTM, BERT, or RoBERTa transformers
- Build a web-based interface using Flask or Streamlit for real-time predictions
- Add support for URL-based article scraping and auto-classification
- Apply cross-validation and hyperparameter tuning for improved accuracy
- Integrate explainability tools (SHAP, LIME) to interpret model predictions

---

## References

- Scikit-learn Documentation: https://scikit-learn.org
- NLTK Documentation: https://www.nltk.org
- TF-IDF Explained: https://en.wikipedia.org/wiki/Tf-idf
- LIAR Dataset (Fake News Benchmark): https://huggingface.co/datasets/liar
- FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet

---

*Fake News Detection using Machine Learning | Python & scikit-learn*
