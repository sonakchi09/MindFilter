// ================================================================
// MindFilter — Content Script
// ================================================================
console.log("MindFilter extension loaded on:", window.location.hostname);

// ── Talk to Flask server through the background script ──────────
// (direct fetch is blocked by Chrome's Private Network Access policy)

function checkServerHealth() {
  chrome.runtime.sendMessage({ type: "HEALTH_CHECK" }, (response) => {
    if (response && response.success) {
      console.log("✅ MindFilter server connected:", response.data.status);
      testAnalyze();
    } else {
      console.error("❌ MindFilter server not reachable:", response?.error);
      console.log("Make sure your Flask server is running: python server/app.py");
    }
  });
}

function testAnalyze() {
  chrome.runtime.sendMessage(
    {
      type: "ANALYZE",
      text: "Everyone else seems to have their life together except me"
    },
    (response) => {
      if (response && response.success) {
        console.log("✅ Test analysis result:", response.data);
      } else {
        console.error("❌ Analysis failed:", response?.error);
      }
    }
  );
}

// ── Run on page load ─────────────────────────────────────────────
console.log("🧠 MindFilter initializing...");
checkServerHealth();