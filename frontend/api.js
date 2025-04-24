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