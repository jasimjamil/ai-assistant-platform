// Example frontend code to interact with your backend API
async function sendMessageToAgent(message, agentId) {
  try {
    const response = await fetch('/api/generate', {
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
    const response = await fetch('/api/status');
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

// API management functions
const API_BASE_URL = ''; // Empty for relative URLs, add domain if needed

// Generic API request function
async function apiRequest(endpoint, method = 'GET', data = null) {
    const url = `${API_BASE_URL}/api${endpoint}`;
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    // Add authentication if available
    const token = localStorage.getItem('auth_token');
    if (token) {
        options.headers['Authorization'] = `Bearer ${token}`;
    }
    
    const response = await fetch(url, options);
    
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Error: ${response.status}`);
    }
    
    // Handle cases where response might not be JSON
    if (response.headers.get('content-type')?.includes('application/json')) {
        return await response.json();
    } else {
        return await response.text();
    }
}

// Status check
async function checkApiStatus() {
    try {
        const result = await apiRequest('/status');
        return result;
    } catch (error) {
        console.error('API status check failed:', error);
        return { status: 'error', message: error.message };
    }
} 