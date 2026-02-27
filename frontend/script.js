const chatWindow = document.getElementById('chatWindow');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

// Function to add a message
function addMessage(text, sender) {
  const msg = document.createElement('div');
  msg.classList.add('message', sender);
  msg.textContent = text;
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight; // auto-scroll
}

// Function to call backend API
async function botResponse(userMsg) {
  addMessage("🤖 Thinking...", "bot"); // temporary message while fetching

  try {
    const res = await fetch("/api/query", {      // ✅ use deployed backend
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question: userMsg })
    });

    const data = await res.json();

    // Remove the "Thinking..." placeholder
    const placeholder = chatWindow.querySelector(".message.bot:last-child");
    if (placeholder) placeholder.remove();

    addMessage(data.answer, "bot");
  } catch (err) {
    console.error(err);
    const placeholder = chatWindow.querySelector(".message.bot:last-child");
    if (placeholder) placeholder.remove();
    addMessage("❌ Sorry, something went wrong!", "bot");
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
}

// Send message
sendBtn.addEventListener('click', () => {
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  chatInput.value = "";
  chatInput.disabled = true;
  botResponse(text);
});

// Enter key to send
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendBtn.click();
});
