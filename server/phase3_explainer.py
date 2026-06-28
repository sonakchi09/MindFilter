# ================================================================
# MindFilter — Phase 3: NLP + LLM Explainer (DistilBERT)
# ================================================================
# In Phase 1 and 2 we built models from scratch.
# In Phase 3 we use a PRETRAINED model — DistilBERT.
#
# DistilBERT was trained by Google/HuggingFace on billions of
# sentences from the internet. It already understands:
# - Word context ("I don't feel happy" ≠ "I feel happy")
# - Sentence structure
# - Subtle emotional language
#
# We're not training it from scratch — we're using a version
# that someone already fine-tuned on emotion data.
# This is called "inference" — running a pretrained model.
#
# This is how most real AI products work today.
# ================================================================

from transformers import pipeline
import json

# ── Severity mapping ─────────────────────────────────────────────
SEVERITY = {
    "sadness" : "HIGH",
    "anger"   : "HIGH",
    "fear"    : "MEDIUM",
    "disgust" : "MEDIUM",
    "surprise": "LOW",
    "love"    : "LOW",
    "joy"     : "LOW"
}

# ── Mental health impact explanations ───────────────────────────
EXPLANATIONS = {
    "sadness": (
        "This post contains language linked to sadness or feeling down. "
        "Repeated exposure to this type of content can reinforce negative "
        "self-perception and make you feel worse about your own life."
    ),
    "anger": (
        "This post contains language that may trigger anger or frustration. "
        "Consuming angry content regularly raises cortisol levels and "
        "keeps your nervous system in a stressed state."
    ),
    "fear": (
        "This post contains anxious or fear-based language. "
        "If you're already stressed, content like this amplifies "
        "those feelings and can make everyday things feel more threatening."
    ),
    "disgust": (
        "This post contains content that may trigger feelings of disgust "
        "or moral outrage. This type of content is highly engaging but "
        "leaves most people feeling drained and negative."
    ),
    "joy": (
        "This post has a genuinely positive emotional tone. "
        "It's unlikely to negatively impact your mental health. "
        "Positive content like this is actually good to consume."
    ),
    "love": (
        "This post contains warm, affectionate language. "
        "Generally positive for mental health, though comparison-driven "
        "love content (e.g. 'perfect couple' posts) can sometimes backfire."
    ),
    "surprise": (
        "This post contains surprising or unexpected content. "
        "Usually neutral, but shocking news or unexpected negative "
        "content can spike anxiety even briefly."
    )
}

# ── Reframe suggestions ──────────────────────────────────────────
REFRAMES = {
    "sadness": "Remember: social media shows highlight reels, not real life. Everyone struggles — most just don't post about it.",
    "anger":   "Consider: online outrage is designed to keep you scrolling. This feeling is real but the content is engineered to trigger it.",
    "fear":    "Note: anxiety-inducing content spreads faster than reassuring content. What you're seeing may not reflect reality accurately.",
    "disgust": "Consider taking a break. Outrage content is the most addictive type of social media content by design.",
    "joy":     "This content seems genuinely positive. Enjoy it!",
    "love":    "Genuine connection content is healthy. Just watch out for idealized relationship content that sets unrealistic standards.",
    "surprise":"Surprising content can be harmless fun. If it's shocking news, verify it before reacting."
}


class MindFilterNLP:
    def __init__(self):
        print("Loading DistilBERT emotion model...")
        print("(Downloading ~250MB model on first run — please wait)")
        
        # This downloads and loads a DistilBERT model fine-tuned
        # specifically on emotion classification
        # "pipeline" is HuggingFace's easy interface for common tasks
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True  # give us scores for ALL emotions
        )
        print("DistilBERT loaded and ready!")
    
    def analyze(self, text):
        """
        Full analysis of a social media post.
        Returns emotion, confidence, severity, explanation, reframe.
        """
        # Truncate very long posts (DistilBERT has a 512 token limit)
        if len(text) > 500:
            text = text[:500]
        
        # Run through DistilBERT
        raw = self.emotion_pipeline(text)
        results = raw[0] if isinstance(raw[0], list) else raw
        
        # Results is a list of {label, score} for each emotion
        # Sort by score to find the top emotion
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        
        top_emotion    = results_sorted[0]["label"].lower()
        confidence     = round(results_sorted[0]["score"] * 100, 2)
        
        # Build full scores dict for transparency
        all_scores = {
            r["label"].lower(): round(r["score"] * 100, 2)
            for r in results_sorted
        }
        
        severity    = SEVERITY.get(top_emotion, "LOW")
        explanation = EXPLANATIONS.get(top_emotion, "")
        reframe     = REFRAMES.get(top_emotion, "")
        
        return {
            "text"       : text,
            "emotion"    : top_emotion,
            "confidence" : confidence,
            "severity"   : severity,
            "all_scores" : all_scores,
            "explanation": explanation,
            "reframe"    : reframe
        }
    
    def analyze_batch(self, posts):
        """Analyze multiple posts and return pattern summary"""
        results = [self.analyze(post) for post in posts]
        
        # Count severity
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        emotion_counts  = {}
        
        for r in results:
            severity_counts[r["severity"]] += 1
            emotion_counts[r["emotion"]] = emotion_counts.get(r["emotion"], 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        total            = len(results)
        
        # Health score
        health_score = round(
            (severity_counts["LOW"] * 1.0 +
             severity_counts["MEDIUM"] * 0.5) / total * 100
        )
        
        return {
            "posts"           : results,
            "total"           : total,
            "severity_counts" : severity_counts,
            "emotion_counts"  : emotion_counts,
            "dominant_emotion": dominant_emotion,
            "health_score"    : health_score
        }


# ── Main: test it ────────────────────────────────────────────────
if __name__ == "__main__":
    
    nlp = MindFilterNLP()
    
    print("\n" + "="*55)
    print("TESTING DISTILBERT ON SOCIAL MEDIA POSTS")
    print("="*55)
    
    test_posts = [
        "Everyone else seems to have their life together except me",
        "Just got into my dream college I can't believe it!!",
        "Why do I always get left out of everything",
        "I'm so scared about my results tomorrow",
        "I can't believe they did that to me, I'm so angry",
        "Feeling so grateful for everything I have today"
    ]
    
    for post in test_posts:
        result = nlp.analyze(post)
        
        print(f"\nPost       : {result['text']}")
        print(f"Emotion    : {result['emotion']} ({result['confidence']}%)")
        print(f"Severity   : {result['severity']}")
        print(f"Explanation: {result['explanation'][:80]}...")
        print(f"Reframe    : {result['reframe'][:80]}...")
        print(f"All scores : {result['all_scores']}")