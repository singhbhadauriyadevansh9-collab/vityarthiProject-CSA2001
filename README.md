# vityarthiProject-CSA2001

# Fake News Detection using Machine Learning

A complete machine learning project that detects whether a news article is **REAL** or **FAKE** using Natural Language Processing (NLP) and classification algorithms.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [ML Pipeline](#ml-pipeline)
- [Models Used](#models-used)
- [Results](#results)
- [Output Files](#output-files)
- [How to Predict on New Articles](#how-to-predict-on-new-articles)
- [Future Improvements](#future-improvements)

---

## Overview

Fake news is a major problem in today's digital world. This project builds a machine learning classifier that can automatically identify whether a given news article is real or fake based on its text content.

The system:
- Preprocesses raw news text using NLP techniques
- Converts text into numerical features using TF-IDF vectorization
- Trains and compares 4 different ML classifiers
- Selects the best-performing model automatically
- Saves the trained model for future predictions

---

## Project Structure

```
fake-news-detection/
│
├── fake_news_detection.py     # Main ML pipeline script
├── fake_news_dataset.csv      # Labeled dataset (100 articles)
├── fake_news_model.pkl        # Saved trained model (generated on run)
├── fake_news_analysis.png     # Visualization charts (generated on run)
└── README.md                  # This file
```

---

## Dataset

The dataset `fake_news_dataset.csv` contains **100 manually labeled news articles** with the following columns:

| Column  | Description                          |
|---------|--------------------------------------|
| `id`    | Unique article identifier            |
| `title` | Headline of the news article         |
| `text`  | Full body text of the article        |
| `label` | Ground truth label: `REAL` or `FAKE` |

- **50 REAL** articles — sourced from credible scientific, economic, and geopolitical events
- **50 FAKE** articles — covering common misinformation patterns (conspiracy theories, health hoaxes, political fabrications)

---

## Technologies Used

| Library        | Purpose                              |
|----------------|--------------------------------------|
| `pandas`       | Data loading and manipulation        |
| `numpy`        | Numerical operations                 |
| `scikit-learn` | ML models, TF-IDF, evaluation        |
| `nltk`         | Stopword removal, stemming           |
| `matplotlib`   | Plotting and visualization           |
| `seaborn`      | Heatmap for confusion matrix         |
| `pickle`       | Saving and loading the trained model |

---

## Installation

Make sure you have Python 3.7+ installed. Then install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

---

## How to Run

1. Place both files in the **same folder**:
   - `fake_news_detection.py`
   - `fake_news_dataset.csv`

2. Run the script:

```bash
python fake_news_detection.py
```

The script will automatically:
- Load and explore the dataset
- Preprocess all text
- Train 4 ML models
- Print evaluation results
- Generate and save visualization charts
- Save the best model as a `.pkl` file

---

## ML Pipeline

```
Raw Text
   |
   v
[Preprocessing]
   - Lowercase conversion
   - Remove URLs, numbers, punctuation
   - Remove English stopwords (NLTK)
   - Porter Stemming
   |
   v
[TF-IDF Vectorization]
   - max_features = 5000
   - ngram_range  = (1, 2)   <-- unigrams + bigrams
   - min_df       = 2
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
   - Accuracy Score
   - Classification Report (Precision, Recall, F1)
   - Confusion Matrix
   |
   v
[Best Model Selected & Saved]
```

---

## Models Used

### 1. Logistic Regression
A linear classifier that estimates the probability of each class. Works very well with TF-IDF features and high-dimensional sparse data.

### 2. Passive Aggressive Classifier
An online learning algorithm designed for large-scale text classification. It "passively" ignores correctly classified samples and "aggressively" updates on misclassified ones.

### 3. Naive Bayes (Multinomial)
A probabilistic classifier based on Bayes' theorem. Assumes feature independence and works exceptionally well for text data — fast and lightweight.

### 4. Random Forest
An ensemble of decision trees. More robust to overfitting and handles non-linear patterns. Slower to train but often achieves high accuracy.

---

## Results

After training, the script automatically picks the best model based on accuracy and prints:

```
Classification Report:
              precision    recall  f1-score   support

        FAKE       x.xx      x.xx      x.xx        10
        REAL       x.xx      x.xx      x.xx        10

    accuracy                           x.xx        20

Confusion Matrix:
[[TP  FP]
 [FN  TN]]
```

Four charts are also generated and saved as `fake_news_analysis.png`:

- **Label Distribution** — Bar chart of REAL vs FAKE counts
- **Model Accuracy Comparison** — Horizontal bar chart for all 4 models
- **Confusion Matrix** — Heatmap of predictions vs actual labels
- **Article Length Distribution** — Histogram comparing length of REAL vs FAKE articles

---

## Output Files

| File                      | Description                              |
|---------------------------|------------------------------------------|
| `fake_news_model.pkl`     | Serialized trained model + vectorizer    |
| `fake_news_analysis.png`  | 4-panel visualization chart              |

---

## How to Predict on New Articles

After running the script, load the saved model to predict any new article:

```python
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
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

# Example
article = "Scientists confirm new vaccine achieves 95% efficacy in clinical trials."
print(predict(article))  # Output: REAL
```

---

## Future Improvements

- Use a larger real-world dataset (e.g., LIAR dataset, FakeNewsNet)
- Add deep learning models (LSTM, BERT, RoBERTa)
- Build a web interface using Flask or Streamlit
- Add URL scraping to classify articles directly from links
- Implement cross-validation for more reliable accuracy estimates
- Add explainability using SHAP or LIME

---

## License

This project is open-source and free to use for educational purposes.

---

## Author

Fake News Detection ML Project  
Built with Python, scikit-learn, and NLTK
