# ================================================================
# MindFilter — Phase 2: Neural Network Emotion Detector
# ================================================================
# In Phase 1 we used Logistic Regression — a simple ML model.
# Now we build a proper neural network using PyTorch.
#
# Same task: classify social media posts into 6 emotions.
# But the neural network learns DEEPER patterns in the text.
#
# HOW A NEURAL NETWORK WORKS (simply):
# - Input layer  → receives the TF-IDF numbers
# - Hidden layers → find patterns humans can't hand-code
# - Output layer  → outputs a probability for each emotion
#
# The network starts with random weights and slowly adjusts
# them using "backpropagation" until it gets good at the task.
# Each full pass through the data is called an "epoch".
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ── Emotion labels (same as Phase 1) ────────────────────────────
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

SEVERITY = {
    "sadness" : "HIGH",
    "anger"   : "HIGH",
    "fear"    : "MEDIUM",
    "surprise": "LOW",
    "love"    : "LOW",
    "joy"     : "LOW"
}

# ================================================================
# STEP 1: DATASET CLASS
# PyTorch needs data wrapped in a Dataset object.
# This tells PyTorch how to access each example one by one.
# ================================================================

class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        # Convert sparse TF-IDF matrix to dense PyTorch tensors
        # A tensor is just PyTorch's version of a numpy array
        self.features = torch.FloatTensor(features.toarray())
        self.labels   = torch.LongTensor(labels)
    
    def __len__(self):
        # How many examples do we have?
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return one example at a time
        return self.features[idx], self.labels[idx]


# ================================================================
# STEP 2: NEURAL NETWORK ARCHITECTURE
# This defines the STRUCTURE of our network —
# how many layers, how many neurons, what activation functions.
# ================================================================

class EmotionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super(EmotionNet, self).__init__()
        
        # ── What is each layer doing? ────────────────────────────
        #
        # Layer 1 (input → hidden1):
        # Takes 10,000 TF-IDF numbers, compresses to 512 neurons
        # Learns broad patterns like "negative words" vs "positive words"
        #
        # Layer 2 (hidden1 → hidden2):
        # Compresses 512 → 256 neurons
        # Learns more specific patterns like "social comparison language"
        #
        # Layer 3 (hidden2 → hidden3):
        # Compresses 256 → 128 neurons
        # Learns even finer patterns
        #
        # Output layer (hidden3 → 6):
        # Outputs a score for each of the 6 emotions
        # The highest score = predicted emotion
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 512),
            nn.ReLU(),        # activation: makes non-linear patterns possible
            nn.Dropout(dropout),  # randomly turns off 30% of neurons during training
                                  # this FORCES the network to not rely on any one neuron
                                  # result: better generalization, less overfitting
            
            # Layer 2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(128, num_classes)
            # No activation here — CrossEntropyLoss handles that internally
        )
    
    def forward(self, x):
        # This defines what happens when data flows THROUGH the network
        # x enters → passes through all layers → outputs 6 scores
        return self.network(x)


# ================================================================
# STEP 3: TRAINING LOOP
# This is where the network actually LEARNS.
# ================================================================

def train_neural_net(train_loader, model, criterion, optimizer, epoch):
    """Run one epoch of training"""
    model.train()  # put model in training mode (enables dropout)
    total_loss = 0
    correct    = 0
    total      = 0
    
    for batch_features, batch_labels in train_loader:
        # ── Forward pass ─────────────────────────────────────────
        # Send data through the network, get predictions
        outputs = model(batch_features)
        
        # ── Calculate loss ────────────────────────────────────────
        # Loss = how wrong the network is right now
        # CrossEntropyLoss compares predictions vs real labels
        # Lower loss = better predictions
        loss = criterion(outputs, batch_labels)
        
        # ── Backward pass (backpropagation) ──────────────────────
        # This is the magic of neural networks.
        # PyTorch figures out HOW to adjust every weight
        # to make the loss smaller next time.
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()        # compute gradients
        optimizer.step()       # update weights
        
        # Track stats
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total   += batch_labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def evaluate_neural_net(test_loader, model):
    """Evaluate model on test data"""
    model.eval()  # put model in evaluation mode (disables dropout)
    all_predictions = []
    all_labels      = []
    
    with torch.no_grad():  # don't compute gradients during evaluation
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_labels.extend(batch_labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    return accuracy, all_predictions, all_labels


# ================================================================
# STEP 4: MAIN — PUT IT ALL TOGETHER
# ================================================================

def main():
    print("=" * 55)
    print("MindFilter — Phase 2: Neural Network")
    print("=" * 55)
    
    # ── Load data ────────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset      = load_dataset("dair-ai/emotion", "split")
    train_texts  = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts   = dataset["test"]["text"]
    test_labels  = dataset["test"]["label"]
    print(f"Loaded {len(train_texts)} training examples")
    
    # ── TF-IDF (same as Phase 1) ─────────────────────────────────
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    print(f"Feature size: {X_train.shape[1]}")
    
    # ── Create PyTorch datasets and dataloaders ──────────────────
    # DataLoader batches the data — instead of training on
    # one example at a time, we train on 32 at once (a "batch")
    # This is much faster and more stable
    train_dataset = EmotionDataset(X_train, list(train_labels))
    test_dataset  = EmotionDataset(X_test,  list(test_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    
    # ── Build the network ────────────────────────────────────────
    print("\nBuilding neural network...")
    model = EmotionNet(
        input_size  = 10000,  # size of TF-IDF vector
        hidden_size = 512,    # neurons in first hidden layer
        num_classes = 6,      # 6 emotions
        dropout     = 0.3     # 30% dropout
    )
    print(model)
    
    # ── Loss function and optimizer ──────────────────────────────
    # CrossEntropyLoss: standard loss for classification tasks
    # Adam optimizer: smart version of gradient descent
    #   it adjusts the learning rate automatically per parameter
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ── Training loop ────────────────────────────────────────────
    print("\nTraining neural network...")
    print("Watch the loss drop and accuracy rise each epoch!\n")
    
    EPOCHS    = 15
    losses    = []
    accuracies = []
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_neural_net(
            train_loader, model, criterion, optimizer, epoch
        )
        test_acc, _, _ = evaluate_neural_net(test_loader, model)
        
        losses.append(train_loss)
        accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Test Acc: {test_acc:.1f}%")
    
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # ── Plot the learning curve ──────────────────────────────────
    # This graph shows the network LEARNING over time
    # Loss going DOWN = network getting better
    # Accuracy going UP = network getting better
    print("\nGenerating learning curve...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(range(1, EPOCHS+1), losses, 'b-o', linewidth=2)
    ax1.set_title("Training Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    ax2.plot(range(1, EPOCHS+1), accuracies, 'g-o', linewidth=2)
    ax2.set_title("Test Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("server/learning_curve.png")
    print("Learning curve saved to server/learning_curve.png")
    plt.show()
    
    # ── Save the model ───────────────────────────────────────────
    print("\nSaving neural network...")
    torch.save({
        "model_state" : model.state_dict(),
        "input_size"  : 10000,
        "hidden_size" : 512,
        "num_classes" : 6,
        "dropout"     : 0.3
    }, "server/phase2_model.pt")
    
    # Save vectorizer (same one we'll reuse)
    with open("server/phase2_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("Model saved to server/phase2_model.pt")
    print("Vectorizer saved to server/phase2_vectorizer.pkl")
    
    # ── Test with real posts ─────────────────────────────────────
    print("\n" + "="*55)
    print("TESTING WITH REAL SOCIAL MEDIA POSTS")
    print("="*55)
    
    test_posts = [
        "Everyone else seems to have their life together except me",
        "Just got into my dream college I can't believe it!!",
        "Why do I always get left out of everything",
        "I'm so scared about my results tomorrow",
        "I can't believe they did that to me, I'm so angry"
    ]
    
    model.eval()
    for post in test_posts:
        vec    = vectorizer.transform([post])
        tensor = torch.FloatTensor(vec.toarray())
        
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        emotion   = EMOTION_LABELS[predicted.item()]
        conf      = confidence.item() * 100
        severity  = SEVERITY[emotion]
        
        print(f"\nPost     : {post}")
        print(f"Emotion  : {emotion} ({conf:.1f}%)")
        print(f"Severity : {severity}")


if __name__ == "__main__":
    main()