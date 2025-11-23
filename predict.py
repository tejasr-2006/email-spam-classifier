import os
import joblib

MODEL_PATH = "models/spam_classifier.joblib"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train it first using spam_classifier.py"
        )
    model = joblib.load(MODEL_PATH)
    return model


def predict_message(message: str) -> str:
    model = load_model()
    pred = model.predict([message])[0]  # "spam" or "ham"
    return pred


if __name__ == "__main__":
    model = load_model()
    print("Spam classifier ready. Type a message (or 'q' to quit).")

    while True:
        msg = input("\nEnter message: ")
        if msg.lower() in {"q", "quit", "exit"}:
            break

        label = model.predict([msg])[0]
        print(f"Prediction: {label}")
