# SMS Spam Detection using NLP

A machine learning web app that detects whether an SMS message is Spam or Not Spam, built with Python, scikit-learn, and Streamlit.

## Project Structure

```
spam_detection_nlp/
│
├── frontend.py                  # Streamlit web app
├── project_spam.ipynb           # Model training notebook
├── spam_model.joblib            # Trained model (generated after running notebook)
├── tfidf_vectorizer.joblib      # Fitted TF-IDF vectorizer (generated after running notebook)
├── SMSSpamCollection            # Dataset
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/spam-detection-nlp.git
cd spam-detection-nlp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### 4. Train the model
Open and run all cells in `project_spam.ipynb`. This will generate:
- `spam_model.joblib`
- `tfidf_vectorizer.joblib`

### 5. Run the app
```bash
streamlit run frontend.py
```

## How It Works

1. Raw SMS text is cleaned (lowercased, punctuation removed)
2. Stopwords are removed and words are lemmatized
3. Text is vectorized using TF-IDF (top 3000 features)
4. A trained ensemble classifier predicts Spam or Not Spam

## Models Trained

| Model | Test Accuracy |
|-------|--------------|
| LinearSVC | ~98.2% |
| Logistic Regression | ~98.0% |
| Random Forest | ~98.0% |
| Multinomial Naive Bayes | ~97.8% |
| Ensemble (Voting) | ~98.0% |

The best model is selected automatically based on cross-validation accuracy.

## Dataset

[SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — 5572 SMS messages labeled as spam or ham (not spam).