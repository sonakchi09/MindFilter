# ================================================================
# MindFilter — Flask Server
# ================================================================
# This is the brain of our backend.
# It's a web server that listens for requests from the
# Chrome extension, passes text through our AI models,
# and sends back the results.
#
# Right now it uses Phase 1 (ML classifier).
# Later we'll add Phase 2, 3, and 4 as extra routes.
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import json
from datetime import datetime

# Import our classifier functions
from phase1_classifier import predict, EMOTION_LABELS, SEVERITY

# ── Create Flask app ─────────────────────────────────────────────
# Flask is a lightweight web framework.
# Think of it as a waiter — it takes requests, passes them to
# the kitchen (our AI), and brings back the result.
app = Flask(__name__)

# CORS allows our Chrome extension (a different "origin")
# to talk to this server. Without this, the browser blocks it.
CORS(app)

# ── Load the trained model ───────────────────────────────────────
# We saved our model to model.pkl in Day 2.
# Now we load it once when the server starts — not on every request.
# This makes responses fast.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

print("Loading emotion model...")
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
    model      = saved["model"]
    vectorizer = saved["vectorizer"]
print("Model loaded and ready!")

# ── Session storage ──────────────────────────────────────────────
# We store each session's analyzed posts in memory.
# A session = one sitting of scrolling social media.
current_session = []

# ================================================================
# ROUTES
# A route is a URL the server listens to.
# When a request hits that URL, the matching function runs.
# ================================================================

@app.route("/", methods=["GET"])
def home():
    """Simple health check — confirms the server is running"""
    return jsonify({
        "status" : "MindFilter server is running",
        "version": "1.0",
        "routes" : ["/analyze", "/session", "/session/summary"]
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main route — takes a social media post and returns:
    - emotion
    - confidence
    - severity
    - explanation
    - session warning if pattern detected
    """
    # Get the text from the request body
    data = request.get_json()
    
    # Validate — make sure text was actually sent
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"].strip()
    
    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400
    
    # Run through our Phase 1 classifier
    result = predict(text, model, vectorizer)
    
    # Add a plain English explanation
    result["explanation"] = generate_explanation(
        result["emotion"],
        result["severity"]
    )
    
    # Add timestamp
    result["timestamp"] = datetime.now().isoformat()
    
    # Track in current session
    current_session.append(result)
    
    # Check for session warning
    warning = check_session_warning(current_session)
    if warning:
        result["warning"] = warning
    
    return jsonify(result)


@app.route("/session", methods=["GET"])
def get_session():
    """Returns everything analyzed in the current session"""
    return jsonify({
        "total_posts" : len(current_session),
        "posts"       : current_session
    })


@app.route("/session/summary", methods=["GET"])
def session_summary():
    """Returns a summary of the current session"""
    if not current_session:
        return jsonify({"message": "No posts analyzed yet"})
    
    # Count emotions
    emotion_counts = {}
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    
    for post in current_session:
        emotion = post["emotion"]
        severity = post["severity"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        severity_counts[severity] += 1
    
    # Find dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # Overall session health score (0-100, higher = healthier)
    total = len(current_session)
    health_score = round(
        ((severity_counts["LOW"] * 1.0 +
          severity_counts["MEDIUM"] * 0.5 +
          severity_counts["HIGH"] * 0.0) / total) * 100
    )
    
    return jsonify({
        "total_posts"     : total,
        "emotion_counts"  : emotion_counts,
        "dominant_emotion": dominant_emotion,
        "severity_counts" : severity_counts,
        "health_score"    : health_score,
        "message"         : get_health_message(health_score)
    })


@app.route("/session/reset", methods=["POST"])
def reset_session():
    """Clears the current session — called when user starts fresh"""
    global current_session
    current_session = []
    return jsonify({"message": "Session reset"})


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def generate_explanation(emotion, severity):
    """
    Returns a plain English explanation of why a post
    might affect mental health — written for a teenager
    """
    explanations = {
        "sadness": "This post contains language associated with sadness or feeling down. Consuming too much content like this can reinforce negative feelings about yourself or your life.",
        "anger":   "This post contains language that may trigger feelings of anger or frustration. Repeated exposure to angry content raises stress levels over time.",
        "fear":    "This post contains language linked to anxiety or worry. If you're already stressed, content like this can amplify those feelings.",
        "joy":     "This post has a positive emotional tone. It's unlikely to negatively impact your mental health.",
        "love":    "This post contains warm, affectionate language. Generally safe for your mental health.",
        "surprise":"This post contains surprising or unexpected content. Usually neutral, but watch out if it's shocking news."
    }
    return explanations.get(emotion, "Unable to generate explanation.")


def check_session_warning(session):
    """
    Checks if the last N posts show a harmful pattern
    and returns a warning message if so
    """
    if len(session) < 5:
        return None
    
    # Look at last 10 posts (or all if less than 10)
    recent = session[-10:]
    high_count = sum(1 for p in recent if p["severity"] == "HIGH")
    
    if high_count >= 7:
        return f"⚠️ {high_count} of your last {len(recent)} posts were high severity. Consider taking a break."
    elif high_count >= 5:
        return f"⚠️ You've been consuming a lot of negative content. {high_count} of your last {len(recent)} posts were high severity."
    
    return None


def get_health_message(score):
    """Returns a human friendly message based on health score"""
    if score >= 80:
        return "Your session looks healthy! Most content you consumed was positive."
    elif score >= 60:
        return "Moderate session. Some negative content but mostly okay."
    elif score >= 40:
        return "This session had quite a bit of negative content. Consider a break."
    else:
        return "This was a very negative session. We strongly recommend stepping away for a bit."


# ── Start the server ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)