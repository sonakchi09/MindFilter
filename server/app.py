# ================================================================
# MindFilter — Main Flask Server (All Phases Connected)
# ================================================================
# This is the central hub that connects everything:
#
# POST /analyze      → Phase 1 (fast ML classifier)
# POST /analyze/deep → Phase 3 (DistilBERT, slower but smarter)
# POST /feedback     → Phase 4 (RL agent learns from feedback)
# GET  /session      → current session data
# GET  /session/summary → session health report
# GET  /risk-profile → what RL agent learned about this user
# POST /session/reset → clear session
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import json
from datetime import datetime

# Import all our AI modules
from phase1_classifier import predict, EMOTION_LABELS, SEVERITY
from phase3_explainer   import MindFilterNLP
from phase4_rl_agent    import MindFilterRLAgent, CONTENT_TYPES

app = Flask(__name__)
CORS(app)

# ================================================================
# LOAD ALL MODELS ON STARTUP
# We load everything once when the server starts.
# This means the first request is slower but every
# subsequent request is fast.
# ================================================================

print("\n" + "="*55)
print("MindFilter Server — Loading AI Models")
print("="*55)

# Phase 1 — ML Classifier (fast)
print("\n[1/3] Loading Phase 1 ML classifier...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    saved      = pickle.load(f)
    ml_model   = saved["model"]
    vectorizer = saved["vectorizer"]
print("      Phase 1 ready!")

# Phase 3 — DistilBERT NLP (smart)
print("\n[2/3] Loading Phase 3 DistilBERT...")
nlp_model = MindFilterNLP()
print("      Phase 3 ready!")

# Phase 4 — RL Agent (personal)
print("\n[3/3] Loading Phase 4 RL Agent...")
rl_agent = MindFilterRLAgent()
print("      Phase 4 ready!")

print("\n" + "="*55)
print("All models loaded! Server starting...")
print("="*55 + "\n")

# ── Session storage ──────────────────────────────────────────────
current_session = []


# ================================================================
# ROUTES
# ================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status" : "MindFilter server running — all models loaded",
        "models" : ["Phase1-ML", "Phase3-DistilBERT", "Phase4-RL"],
        "routes" : [
            "POST /analyze",
            "POST /analyze/deep",
            "POST /feedback",
            "GET  /session",
            "GET  /session/summary",
            "GET  /risk-profile",
            "POST /session/reset"
        ]
    })


@app.route("/analyze", methods=["POST"])
def analyze_fast():
    """
    Fast analysis using Phase 1 ML classifier.
    Use this for real-time analysis as the user scrolls.
    Response time: ~50ms
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    # Phase 1 prediction
    result = predict(text, ml_model, vectorizer)
    result["model"]     = "phase1-ml"
    result["timestamp"] = datetime.now().isoformat()

    # Add explanation
    result["explanation"] = generate_explanation(
        result["emotion"], result["severity"]
    )

    # Track session
    current_session.append(result)

    # Check for warning
    warning = check_session_warning(current_session)
    if warning:
        result["warning"] = warning

    return jsonify(result)


@app.route("/analyze/deep", methods=["POST"])
def analyze_deep():
    """
    Deep analysis using Phase 3 DistilBERT.
    Use this when the user wants a detailed breakdown.
    Response time: ~500ms
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    # Phase 3 deep analysis
    result = nlp_model.analyze(text)
    result["model"]     = "phase3-distilbert"
    result["timestamp"] = datetime.now().isoformat()

    # Track session
    current_session.append(result)

    # Check for warning
    warning = check_session_warning(current_session)
    if warning:
        result["warning"] = warning

    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Receives user feedback on whether a post affected them.
    Feeds directly into the RL agent for learning.

    Expected body:
    {
        "content_type": "fitness_lifestyle",
        "affected": true
    }
    """
    data = request.get_json()
    if not data or "content_type" not in data:
        return jsonify({"error": "content_type required"}), 400

    content_type  = data["content_type"]
    user_affected = data.get("affected", False)

    # Validate content type
    if content_type not in CONTENT_TYPES:
        return jsonify({
            "error"          : "Invalid content type",
            "valid_types"    : CONTENT_TYPES
        }), 400

    # RL agent processes the feedback and learns
    result = rl_agent.process_feedback(content_type, user_affected)

    return jsonify(result)


@app.route("/session", methods=["GET"])
def get_session():
    return jsonify({
        "total_posts": len(current_session),
        "posts"      : current_session
    })


@app.route("/session/summary", methods=["GET"])
def session_summary():
    if not current_session:
        return jsonify({"message": "No posts analyzed yet"})

    emotion_counts  = {}
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for post in current_session:
        emotion  = post["emotion"]
        severity = post["severity"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        severity_counts[severity] += 1

    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    total            = len(current_session)
    health_score     = round(
        ((severity_counts["LOW"] * 1.0 +
          severity_counts["MEDIUM"] * 0.5) / total) * 100
    )

    return jsonify({
        "total_posts"     : total,
        "emotion_counts"  : emotion_counts,
        "dominant_emotion": dominant_emotion,
        "severity_counts" : severity_counts,
        "health_score"    : health_score,
        "message"         : get_health_message(health_score)
    })


@app.route("/risk-profile", methods=["GET"])
def risk_profile():
    """Returns what the RL agent has learned about this user"""
    profile = rl_agent.get_risk_profile()
    episodes = rl_agent.total_episodes

    return jsonify({
        "total_episodes": episodes,
        "epsilon"       : round(rl_agent.epsilon, 3),
        "risk_profile"  : profile,
        "message"       : f"Based on {episodes} interactions with MindFilter"
    })


@app.route("/session/reset", methods=["POST"])
def reset_session():
    global current_session
    current_session = []
    return jsonify({"message": "Session reset"})


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def generate_explanation(emotion, severity):
    explanations = {
        "sadness" : "This post contains language linked to sadness. Repeated exposure can reinforce negative feelings about your own life.",
        "anger"   : "This post may trigger anger or frustration. Angry content raises stress levels and keeps your nervous system activated.",
        "fear"    : "This post contains anxious language. If you're already stressed, content like this amplifies those feelings.",
        "joy"     : "This post has a positive tone. It's unlikely to negatively impact your mental health.",
        "love"    : "This post contains warm language. Generally safe, though idealized content can sometimes backfire.",
        "surprise": "This post contains surprising content. Usually neutral — watch out if it's shocking news."
    }
    return explanations.get(emotion, "")


def check_session_warning(session):
    if len(session) < 5:
        return None
    recent     = session[-10:]
    high_count = sum(1 for p in recent if p["severity"] == "HIGH")
    if high_count >= 7:
        return f"⚠️ {high_count} of your last {len(recent)} posts were high severity. Consider taking a break."
    elif high_count >= 5:
        return f"⚠️ You've been consuming a lot of negative content lately."
    return None


def get_health_message(score):
    if score >= 80:
        return "Your session looks healthy!"
    elif score >= 60:
        return "Moderate session — some negative content but mostly okay."
    elif score >= 40:
        return "Quite a bit of negative content this session. Consider a break."
    else:
        return "Very negative session. We strongly recommend stepping away."


# ── Start server ─────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)