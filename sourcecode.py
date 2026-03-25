"""
============================================================
  FAKE NEWS DETECTION PROJECT
  Using Machine Learning (NLP + Classification)
============================================================
  Features:
  - Data loading and exploration
  - Text preprocessing (cleaning, stemming, TF-IDF)
  - Multiple ML models: Logistic Regression, Naive Bayes,
    Random Forest, Passive Aggressive Classifier
  - Model evaluation with accuracy, confusion matrix, report
  - Prediction on new text
============================================================
"""

# pip install pandas numpy scikit-learn matplotlib seaborn nltk

import sys
import os
import re
import string
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Force UTF-8 output so special chars never crash on Windows
sys.stdout.reconfigure(encoding='utf-8')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

from nltk.corpus import stopwords
from nltk.stem  import PorterStemmer

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes             import MultinomialNB
from sklearn.ensemble                import RandomForestClassifier
from sklearn.metrics                 import (accuracy_score, classification_report,
                                             confusion_matrix)

# Auto-locate the CSV next to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================
# 1. LOAD DATA
# =============================================================
print("=" * 60)
print("  FAKE NEWS DETECTION PROJECT")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE_DIR, "fake_news_dataset.csv"))

print("\n[OK] Dataset loaded successfully!")
print(f"     Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLabel Distribution:")
print(df['label'].value_counts())
print(f"\n  REAL : {(df['label']=='REAL').sum()} articles")
print(f"  FAKE : {(df['label']=='FAKE').sum()} articles")


# =============================================================
# 2. TEXT PREPROCESSING
# =============================================================
print("\n" + "=" * 60)
print("  TEXT PREPROCESSING")
print("=" * 60)

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def preprocess_text(text):
    """Lowercase -> strip URLs/numbers/punct -> remove stopwords -> stem."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['cleaned'] = df['content'].apply(preprocess_text)

print("\n[OK] Text preprocessing complete!")
print("\nSample:")
print(f"  Original : {df['content'].iloc[0][:100]}...")
print(f"  Cleaned  : {df['cleaned'].iloc[0][:100]}...")

df['label_encoded'] = df['label'].map({'REAL': 1, 'FAKE': 0})


# =============================================================
# 3. TRAIN / TEST SPLIT
# =============================================================
X = df['cleaned']
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData Split:")
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")


# =============================================================
# 4. TF-IDF VECTORIZATION
# =============================================================
print("\n" + "=" * 60)
print("  TF-IDF VECTORIZATION")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n[OK] TF-IDF matrix created!")
print(f"  Vocabulary size : {len(tfidf.vocabulary_)}")
print(f"  Train matrix    : {X_train_tfidf.shape}")
print(f"  Test  matrix    : {X_test_tfidf.shape}")


# =============================================================
# 5. TRAIN MULTIPLE MODELS
# =============================================================
print("\n" + "=" * 60)
print("  TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Passive Aggressive"  : PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    "Naive Bayes"         : MultinomialNB(alpha=0.1),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': acc, 'y_pred': y_pred}
    print(f"\n  [OK] {name}")
    print(f"       Accuracy: {acc*100:.2f}%")


# =============================================================
# 6. DETAILED EVALUATION - BEST MODEL
# =============================================================
best_name = max(results, key=lambda k: results[k]['accuracy'])
best      = results[best_name]

print("\n" + "=" * 60)
print(f"  BEST MODEL: {best_name}")
print("=" * 60)

print(f"\n  Accuracy : {best['accuracy']*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'],
                            target_names=['FAKE', 'REAL']))

cm = confusion_matrix(y_test, best['y_pred'])
print("Confusion Matrix:")
print(cm)
print(f"\n  True Positives  (Real -> Real) : {cm[1][1]}")
print(f"  True Negatives  (Fake -> Fake) : {cm[0][0]}")
print(f"  False Positives (Fake -> Real) : {cm[0][1]}")
print(f"  False Negatives (Real -> Fake) : {cm[1][0]}")


# =============================================================
# 7. VISUALIZATIONS
# =============================================================
print("\n" + "=" * 60)
print("  GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fake News Detection - Model Analysis",
             fontsize=16, fontweight='bold')

# Plot 1: Label Distribution
ax1 = axes[0, 0]
label_counts = df['label'].value_counts()
colors = ['#e74c3c', '#2ecc71']
bars = ax1.bar(label_counts.index, label_counts.values,
               color=colors, edgecolor='black', linewidth=0.5)
ax1.set_title("Dataset Label Distribution", fontweight='bold')
ax1.set_xlabel("News Category")
ax1.set_ylabel("Count")
for bar, val in zip(bars, label_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             str(val), ha='center', fontweight='bold')

# Plot 2: Model Accuracy Comparison
ax2 = axes[0, 1]
names      = list(results.keys())
accs       = [results[n]['accuracy']*100 for n in names]
bar_colors = ['#f39c12' if n == best_name else '#3498db' for n in names]
bars2 = ax2.barh(names, accs, color=bar_colors,
                 edgecolor='black', linewidth=0.5)
ax2.set_title("Model Accuracy Comparison", fontweight='bold')
ax2.set_xlabel("Accuracy (%)")
ax2.set_xlim(0, 115)
for bar, val in zip(bars2, accs):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontweight='bold')
ax2.axvline(x=80, color='red', linestyle='--',
            alpha=0.5, label='80% threshold')
ax2.legend()

# Plot 3: Confusion Matrix Heatmap
ax3 = axes[1, 0]
cm_df = pd.DataFrame(cm,
    index=['Actual FAKE', 'Actual REAL'],
    columns=['Predicted FAKE', 'Predicted REAL'])
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
            linewidths=1, linecolor='gray', ax=ax3,
            annot_kws={"size": 14, "weight": "bold"})
ax3.set_title(f"Confusion Matrix\n({best_name})", fontweight='bold')

# Plot 4: Article Length Distribution
ax4 = axes[1, 1]
df['content_length'] = df['content'].apply(len)
ax4.hist(df[df['label']=='FAKE']['content_length'],
         bins=20, alpha=0.6, color='#e74c3c',
         label='FAKE', edgecolor='black')
ax4.hist(df[df['label']=='REAL']['content_length'],
         bins=20, alpha=0.6, color='#2ecc71',
         label='REAL', edgecolor='black')
ax4.set_title("Article Length Distribution", fontweight='bold')
ax4.set_xlabel("Character Count")
ax4.set_ylabel("Frequency")
ax4.legend()

plt.tight_layout()
out_png = os.path.join(BASE_DIR, "fake_news_analysis.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n[OK] Plot saved as '{out_png}'")


# =============================================================
# 8. TOP INDICATOR WORDS
# =============================================================
print("\n" + "=" * 60)
print("  TOP WORDS FOR FAKE vs REAL NEWS")
print("=" * 60)

if hasattr(best['model'], 'coef_'):
    feature_names = tfidf.get_feature_names_out()
    coef          = best['model'].coef_[0]

    print("\nTop 15 words -> FAKE news:")
    for idx in coef.argsort()[:15]:
        print(f"  {feature_names[idx]:<22} score: {coef[idx]:.3f}")

    print("\nTop 15 words -> REAL news:")
    for idx in coef.argsort()[-15:][::-1]:
        print(f"  {feature_names[idx]:<22} score: {coef[idx]:.3f}")


# =============================================================
# 9. PREDICT NEW ARTICLES
# =============================================================
print("\n" + "=" * 60)
print("  PREDICTION ON NEW ARTICLES")
print("=" * 60)

def predict_news(article_text, model, vectorizer):
    """Return (label, confidence%) for a raw news string."""
    cleaned    = preprocess_text(article_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if hasattr(model, 'predict_proba'):
        confidence = max(model.predict_proba(vectorized)[0]) * 100
    elif hasattr(model, 'decision_function'):
        score      = abs(model.decision_function(vectorized)[0])
        confidence = min(score * 30, 99)
    else:
        confidence = None

    label = "[REAL]" if prediction == 1 else "[FAKE]"
    return label, confidence

test_articles = [
    {
        "title": "Test 1 - Likely REAL",
        "text" : ("NASA scientists confirmed today that the James Webb Space Telescope "
                  "has captured the most detailed images of a distant galaxy, providing "
                  "new insights about star formation in the early universe.")
    },
    {
        "title": "Test 2 - Likely FAKE",
        "text" : ("A secret leaked document proves that the government is adding "
                  "mind-control chemicals to tap water to make citizens obedient. "
                  "Thousands of whistleblowers have confirmed this global conspiracy.")
    },
    {
        "title": "Test 3 - Likely REAL",
        "text" : ("The International Monetary Fund released its quarterly economic "
                  "outlook showing that global GDP growth is expected to slow to 2.9 "
                  "percent in 2024 due to high interest rates and geopolitical tensions.")
    },
    {
        "title": "Test 4 - Likely FAKE",
        "text" : ("Drinking lemon juice mixed with baking soda every morning has been "
                  "scientifically proven to completely cure diabetes and cancer within "
                  "21 days. Big Pharma is desperately trying to hide this miracle cure.")
    },
]

print()
for article in test_articles:
    label, confidence = predict_news(article['text'], best['model'], tfidf)
    conf_str = f"  (Confidence: {confidence:.1f}%)" if confidence else ""
    print(f"  {article['title']}")
    print(f"    Prediction: {label}{conf_str}")
    print()


# =============================================================
# 10. SAVE MODEL
# =============================================================
print("=" * 60)
print("  SAVING MODEL")
print("=" * 60)

model_path = os.path.join(BASE_DIR, 'fake_news_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({'model': best['model'], 'vectorizer': tfidf,
                 'model_name': best_name}, f)

print(f"\n[OK] Model saved -> {model_path}")
print(f"     Model type : {best_name}")
print(f"     Accuracy   : {best['accuracy']*100:.2f}%")

print("\n" + "=" * 60)
print("  HOW TO LOAD & USE THE SAVED MODEL")
print("=" * 60)
print("""
  import pickle

  with open('fake_news_model.pkl', 'rb') as f:
      saved = pickle.load(f)

  model      = saved['model']
  vectorizer = saved['vectorizer']

  def predict(text):
      cleaned = preprocess_text(text)
      vec     = vectorizer.transform([cleaned])
      pred    = model.predict(vec)[0]
      return 'REAL' if pred == 1 else 'FAKE'

  print(predict("Your news article text here..."))
""")

print("=" * 60)
print("  PROJECT COMPLETE!")
print("=" * 60)
print("""
  Files generated:
    fake_news_dataset.csv    -> 100 labeled articles
    fake_news_detection.py   -> This ML pipeline
    fake_news_analysis.png   -> Visualization charts
    fake_news_model.pkl      -> Saved trained model

  Models trained:
    Logistic Regression
    Passive Aggressive Classifier
    Naive Bayes
    Random Forest

  Pipeline:
    Load -> Preprocess -> TF-IDF -> Train -> Evaluate -> Predict
""")
