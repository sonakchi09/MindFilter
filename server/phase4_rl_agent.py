# ================================================================
# MindFilter — Phase 4: Reinforcement Learning Agent
# ================================================================
# The previous phases treat everyone the same.
# Phase 4 learns YOUR personal triggers.
#
# FITNESS posts might destroy one person's self image.
# The same posts might motivate another person completely.
#
# HOW REINFORCEMENT LEARNING WORKS:
# - Agent    = our AI that makes decisions
# - State    = what type of content you just consumed
# - Action   = predict whether this will affect you or not
# - Reward   = +1 if prediction was right, -1 if wrong
# - Q-table  = a table the agent updates after every interaction
#
# Over time the Q-table gets better and better at predicting
# which content types affect THIS specific user.
#
# This is Q-learning — the foundation of modern RL.
# ================================================================

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ── Content categories ───────────────────────────────────────────
# These are the types of social media content we track
CONTENT_TYPES = [
    "fitness_lifestyle",    # gym, diet, body image posts
    "academic_pressure",    # grades, college, career posts
    "relationship_content", # couples, dating, friendship posts
    "financial_flex",       # money, luxury, success posts
    "news_politics",        # current events, outrage posts
    "creative_content",     # art, music, entertainment posts
    "motivational",         # hustle culture, inspiration posts
    "personal_update"       # life updates, celebrations posts
]

# ── Actions the agent can take ───────────────────────────────────
# 0 = predict "this will NOT affect the user"
# 1 = predict "this WILL affect the user negatively"
ACTIONS = [0, 1]
ACTION_LABELS = {0: "safe", 1: "harmful"}

# ── History path ─────────────────────────────────────────────────
HISTORY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "user_history.json"
)


class MindFilterRLAgent:
    def __init__(self, epsilon=0.3, alpha=0.1, gamma=0.9):
        """
        epsilon = exploration rate
                  how often the agent tries NEW predictions
                  vs using what it already learned
                  0.3 = 30% exploration, 70% exploitation
        
        alpha   = learning rate
                  how quickly the agent updates its beliefs
                  0.1 = slow, steady learning
        
        gamma   = discount factor
                  how much future rewards matter vs immediate ones
                  0.9 = future rewards matter a lot
        """
        self.epsilon = epsilon
        self.alpha   = alpha
        self.gamma   = gamma
        
        # ── Q-table ──────────────────────────────────────────────
        # Shape: (8 content types) x (2 actions)
        # Each cell = expected reward for taking that action
        #             in that content type state
        # Starts at zero — agent knows nothing yet
        self.q_table = np.zeros((len(CONTENT_TYPES), len(ACTIONS)))
        
        # Track history for the reward curve
        self.episode_rewards  = []
        self.episode_accuracy = []
        self.total_episodes   = 0
        
        # Load existing history if available
        self.load_history()
    
    def get_state(self, content_type):
        """Convert content type string to state index"""
        if content_type in CONTENT_TYPES:
            return CONTENT_TYPES.index(content_type)
        return 0  # default to first category
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        
        This is how the agent balances:
        EXPLORATION — trying new things to discover what works
        EXPLOITATION — using what it already knows works
        
        If random number < epsilon → EXPLORE (random action)
        Otherwise → EXPLOIT (best action from Q-table)
        """
        if np.random.random() < self.epsilon:
            # Explore: pick a random action
            return np.random.choice(ACTIONS)
        else:
            # Exploit: pick the action with highest Q-value
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Q-learning update rule — the core of RL.
        
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s')) - Q(s,a)]
        
        In plain English:
        "Update my belief about this action based on
         how wrong I was + how good the future looks"
        """
        current_q  = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        # The Q-learning formula
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
    
    def process_feedback(self, content_type, user_affected):
        """
        Called after user gives feedback on whether
        a post affected them or not.
        
        content_type  = what type of post it was
        user_affected = True if it affected them, False if not
        """
        state  = self.get_state(content_type)
        action = self.choose_action(state)
        
        # Did the agent predict correctly?
        predicted_harmful = (action == 1)
        prediction_correct = (predicted_harmful == user_affected)
        
        # Reward: +1 for correct prediction, -1 for wrong
        reward = 1.0 if prediction_correct else -1.0
        
        # Decay epsilon over time — less exploration as we learn more
        self.epsilon = max(0.05, self.epsilon * 0.995)
        
        # Update Q-table
        next_state = (state + 1) % len(CONTENT_TYPES)
        self.update_q_table(state, action, reward, next_state)
        
        # Track episode
        self.total_episodes += 1
        self.episode_rewards.append(reward)
        
        # Running accuracy (last 20 episodes)
        recent = self.episode_rewards[-20:]
        accuracy = sum(1 for r in recent if r > 0) / len(recent) * 100
        self.episode_accuracy.append(accuracy)
        
        # Save updated history
        self.save_history()
        
        return {
            "content_type"      : content_type,
            "agent_predicted"   : ACTION_LABELS[action],
            "user_was_affected" : user_affected,
            "prediction_correct": prediction_correct,
            "reward"            : reward,
            "total_episodes"    : self.total_episodes,
            "recent_accuracy"   : round(accuracy, 1),
            "epsilon"           : round(self.epsilon, 3)
        }
    
    def get_risk_profile(self):
        """
        Returns which content types are most risky for THIS user
        based on what the agent has learned so far.
        """
        risk_profile = {}
        for i, content_type in enumerate(CONTENT_TYPES):
            # High Q-value for action 1 = agent learned this is harmful
            harmful_q = self.q_table[i, 1]
            safe_q    = self.q_table[i, 0]
            
            if harmful_q > safe_q:
                risk = "HIGH RISK"
            elif harmful_q == safe_q:
                risk = "UNKNOWN"
            else:
                risk = "LOW RISK"
            
            risk_profile[content_type] = {
                "risk"    : risk,
                "harmful_q": round(harmful_q, 3),
                "safe_q"  : round(safe_q, 3)
            }
        
        return risk_profile
    
    def plot_learning_curve(self):
        """Show how the agent's accuracy improved over time"""
        if len(self.episode_accuracy) < 5:
            print("Need more episodes to plot learning curve")
            return
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_accuracy, 'purple', linewidth=2)
        plt.axhline(y=50, color='red', linestyle='--',
                    label='Random baseline (50%)')
        plt.title("RL Agent Learning Curve — Prediction Accuracy Over Time")
        plt.xlabel("Episode (each feedback = 1 episode)")
        plt.ylabel("Accuracy % (last 20 episodes)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("server/rl_learning_curve.png")
        print("Learning curve saved to server/rl_learning_curve.png")
        plt.show()
    
    def save_history(self):
        """Save agent state to disk so it persists between sessions"""
        history = {
            "q_table"          : self.q_table.tolist(),
            "episode_rewards"  : self.episode_rewards,
            "episode_accuracy" : self.episode_accuracy,
            "total_episodes"   : self.total_episodes,
            "epsilon"          : self.epsilon,
            "last_updated"     : datetime.now().isoformat()
        }
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
    
    def load_history(self):
        """Load previous agent state if it exists"""
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, "r") as f:
                    history = json.load(f)
                
                if "q_table" in history:
                    self.q_table         = np.array(history["q_table"])
                    self.episode_rewards  = history.get("episode_rewards", [])
                    self.episode_accuracy = history.get("episode_accuracy", [])
                    self.total_episodes   = history.get("total_episodes", 0)
                    self.epsilon          = history.get("epsilon", 0.3)
                    print(f"Loaded existing history — {self.total_episodes} episodes")
            except:
                print("Starting fresh — no valid history found")


# ── Main: simulate a session ─────────────────────────────────────
if __name__ == "__main__":
    
    print("="*55)
    print("MindFilter — Phase 4: RL Agent")
    print("="*55)
    
    agent = MindFilterRLAgent()
    
    # Simulate 50 episodes of user feedback
    # In real use this happens one post at a time
    print("\nSimulating 50 episodes of user feedback...")
    print("(In real use, each episode = one post the user rates)\n")
    
    # Simulated user profile — this person is sensitive to:
    # fitness content and academic pressure, but fine with creative content
    user_sensitive_to = [
        "fitness_lifestyle",
        "academic_pressure",
        "financial_flex"
    ]
    
    import random
    for episode in range(50):
        # Pick a random content type
        content = random.choice(CONTENT_TYPES)
        
        # Simulate whether this user was affected
        # (in real app, this comes from the user clicking a button)
        user_affected = content in user_sensitive_to
        
        # Add some randomness to make it realistic
        if random.random() < 0.15:
            user_affected = not user_affected
        
        result = agent.process_feedback(content, user_affected)
        
        if episode % 10 == 0:
            print(f"Episode {episode+1:3d} | "
                  f"Content: {content:25s} | "
                  f"Affected: {str(user_affected):5s} | "
                  f"Correct: {str(result['prediction_correct']):5s} | "
                  f"Accuracy: {result['recent_accuracy']}%")
    
    print(f"\nFinal accuracy: {agent.episode_accuracy[-1]:.1f}%")
    print(f"Total episodes: {agent.total_episodes}")
    print(f"Epsilon (exploration rate): {agent.epsilon:.3f}")
    
    # Show what the agent learned about this user
    print("\n" + "="*55)
    print("WHAT THE AGENT LEARNED ABOUT THIS USER")
    print("="*55)
    risk_profile = agent.get_risk_profile()
    for content_type, data in risk_profile.items():
        print(f"{content_type:25s} → {data['risk']}")
    
    # Plot the learning curve
    agent.plot_learning_curve()