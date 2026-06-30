// ================================================================
// MindFilter — Background Script
// ================================================================
// Content scripts running on Instagram/Twitter can't directly
// call localhost due to Chrome's Private Network Access policy.
//
// SOLUTION: content.js sends a message to this background script,
// which IS allowed to call localhost, then sends the result back.
// ================================================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ANALYZE") {
    fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message.text })
    })
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    
    return true; // keeps the message channel open for async response
  }
  
  if (message.type === "HEALTH_CHECK") {
    fetch("http://127.0.0.1:5000/")
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    
    return true;
  }
});

console.log("MindFilter background script loaded");