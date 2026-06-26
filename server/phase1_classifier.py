# ================================================================
# MindFilter — Phase 1: Emotion Classifier
# ================================================================
# This file is the brain of MindFilter's first layer.
# It reads a social media post and classifies it into one of
# 6 emotions: joy, sadness, anger, fear, love, surprise
#
# HOW IT WORKS:
# 1. We load a dataset of 20,000 labeled sentences
# 2. We convert text into numbers (TF-IDF)
# 3. We train a Logistic Regression model on those numbers
# 4. We save the model so Flask can use it later
# ================================================================

import pickle
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# ── Emotion labels ──────────────────────────────────────────────
# The dataset uses numbers for emotions. This maps them to words.
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# ── Severity mapping ─────────────────────────────────────────────
# For MindFilter we also want to know HOW harmful each emotion is
# This is the mental health impact score
SEVERITY = {
    "sadness":  "HIGH",
    "anger":    "HIGH",
    "fear":     "MEDIUM",
    "surprise": "LOW",
    "love":     "LOW",
    "joy":      "LOW"
}

def load_data():
    """Load the emotion dataset from HuggingFace"""
    print("Loading emotion dataset...")
    print("(This will take a minute the first time — downloading ~3MB)")
    
    dataset = load_dataset("dair-ai/emotion", "split")
    
    # Extract texts and labels from train/test splits
    train_texts  = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts   = dataset["test"]["text"]
    test_labels  = dataset["test"]["label"]
    
    print(f"Training examples : {len(train_texts)}")
    print(f"Test examples     : {len(test_texts)}")
    print(f"\nSample post  : '{train_texts[0]}'")
    print(f"Sample emotion: {EMOTION_LABELS[train_labels[0]]}")
    
    return train_texts, train_labels, test_texts, test_labels


def train_model(train_texts, train_labels):
    """Convert text to numbers and train the classifier"""
    
    # ── Step 1: TF-IDF Vectorization ────────────────────────────
    # Computers can't read text — they only understand numbers.
    # TF-IDF converts each post into a list of numbers.
    # Words that appear a lot in ONE post but rarely across ALL
    # posts get a HIGH score — they are more meaningful.
    # Common words like "the", "and", "is" get LOW scores.
    print("\nConverting text to numbers with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,   # only keep the 10,000 most important words
        stop_words="english", # ignore common words like "the", "is", "and"
        ngram_range=(1, 2)    # also consider pairs of words e.g. "feel sad"
    )
    
    # fit_transform = learn the vocabulary AND convert training text
    X_train = vectorizer.fit_transform(train_texts)
    print(f"Each post is now a vector of {X_train.shape[1]} numbers")
    
    # ── Step 2: Train Logistic Regression ───────────────────────
    # Logistic Regression learns which words/phrases push
    # the prediction toward each emotion.
    # e.g. "worthless", "alone", "nobody" → sadness
    #      "excited", "amazing", "can't wait" → joy
    print("\nTraining emotion classifier...")
    model = LogisticRegression(
        max_iter=1000,  # maximum times to adjust weights
        C=1.0,          # how strictly to fit training data
        solver="lbfgs", # the math algorithm used to optimize
        
    )
    model.fit(X_train, train_labels)
    print("Training complete!")
    
    return model, vectorizer


def evaluate_model(model, vectorizer, test_texts, test_labels):
    """Test how accurate our model is"""
    print("\nEvaluating model on test data...")
    
    # Transform test texts using the SAME vectorizer
    # (we only call transform here, not fit_transform
    #  because we don't want to learn new vocabulary from test data)
    X_test = vectorizer.transform(test_texts)
    
    # Get predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions) * 100
    
    print(f"\nAccuracy: {accuracy:.2f}%")
    print("\nDetailed breakdown:")
    print(classification_report(
        test_labels,
        predictions,
        target_names=list(EMOTION_LABELS.values())
    ))
    
    return accuracy


def save_model(model, vectorizer):
    """Save model to disk so Flask can load it later"""
    print("\nSaving model...")
    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print("Model saved to model.pkl")


def predict(text, model, vectorizer):
    """
    Given any text, return:
    - emotion label
    - confidence score
    - mental health severity
    """
    # Convert the text to numbers
    vec = vectorizer.transform([text])
    
    # Get prediction and confidence
    prediction   = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    confidence   = max(probabilities) * 100
    
    emotion  = EMOTION_LABELS[prediction]
    severity = SEVERITY[emotion]
    
    return {
        "text"      : text,
        "emotion"   : emotion,
        "confidence": round(confidence, 2),
        "severity"  : severity
    }


# ── Main: run everything ─────────────────────────────────────────
if __name__ == "__main__":
    
    # Step 1: Load data
    train_texts, train_labels, test_texts, test_labels = load_data()
    
    # Step 2: Train
    model, vectorizer = train_model(train_texts, train_labels)
    
    # Step 3: Evaluate
    evaluate_model(model, vectorizer, test_texts, test_labels)
    
    # Step 4: Save
    save_model(model, vectorizer)
    
    # Step 5: Test with real social media style posts
    print("\n" + "="*55)
    print("TESTING WITH REAL SOCIAL MEDIA POSTS")
    print("="*55)
    
    test_posts = [
        "Everyone else seems to have their life together except me",
        "Just got into my dream college I can't believe it!!",
        "Why do I always get left out of everything",
        "I'm so scared about my results tomorrow",
        "Spending the evening with people I love",
        "I can't believe they did that to me, I'm so angry"
    ]
    
    for post in test_posts:
        result = predict(post, model, vectorizer)
        print(f"\nPost      : {result['text']}")
        print(f"Emotion   : {result['emotion']} ({result['confidence']}%)")
        print(f"Severity  : {result['severity']}")