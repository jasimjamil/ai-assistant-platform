// Example frontend code to interact with your backend API
async function sendMessageToAgent(message, agentId) {
  try {
    const response = await fetch('http://localhost:8000/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: message,
        agent_id: agentId
      }),
    });
    
    const data = await response.json();
    
    if (data.error) {
      console.error('Error:', data.error);
      return `Error: ${data.error}`;
    }
    
    return data.response;
  } catch (error) {
    console.error('Network error:', error);
    return 'Sorry, there was a network error connecting to the agent.';
  }
}

// Example function to check backend status
async function checkBackendStatus() {
  try {
    const response = await fetch('http://localhost:8000/api/status');
    const status = await response.json();
    
    console.log('Backend status:', status);
    
    // Update UI based on status
    document.getElementById('status-indicator').className = 
      status.google_api_available ? 'status-online' : 'status-offline';
      
    // Show a warning if Telegram is offline
    if (!status.telegram_available) {
      document.getElementById('telegram-warning').style.display = 'block';
    }
    
    return status;
  } catch (error) {
    console.error('Cannot connect to backend:', error);
    document.getElementById('status-indicator').className = 'status-error';
    return null;
  }
}

// Call the status check when page loads
document.addEventListener('DOMContentLoaded', () => {
  checkBackendStatus();
}); 