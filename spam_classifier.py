import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_PATH = "data/spam.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.joblib")


def load_data(path: str) -> pd.DataFrame:
    # Adjust encoding if needed
    df = pd.read_csv(path, encoding="latin-1")

    # For SMS Spam Collection: keep only label and text columns
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]]
        df = df.rename(columns={"v1": "label", "v2": "text"})
    else:
        # If you use some other dataset, make sure you have 'label' and 'text' columns
        df = df[["label", "text"]]

    # Remove missing values
    df = df.dropna(subset=["label", "text"])

    return df


def train_model(df: pd.DataFrame):
    X = df["text"]
    y = df["label"]

    # Stratified split to keep spam/ham ratio similar in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Pipeline: TF-IDF vectorizer + Multinomial Naive Bayes classifier
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                # You can tweak these:
                ngram_range=(1, 2),  # unigrams + bigrams
                max_df=0.9,
                min_df=2,
            )),
            ("clf", MultinomialNB()),
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    return model, acc


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages.")
    model, acc = train_model(df)
