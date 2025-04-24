// Global variables
const API_URL = window.location.hostname === 'localhost' 
    ? '/api'
    : 'https://your-project-name.vercel.app/api';
let authToken = localStorage.getItem('authToken');
let currentUser = null;

// DOM Elements
const authContainer = document.getElementById('auth-container');
const appContainer = document.querySelector('.app-container');
const loginScreen = document.getElementById('login-screen');
const signupScreen = document.getElementById('signup-screen');
const loginForm = document.getElementById('login-form');
const signupForm = document.getElementById('signup-form');
const dashboardScreen = document.getElementById('dashboard-screen');
const agentsScreen = document.getElementById('agents-screen');
const knowledgeScreen = document.getElementById('knowledge-screen');
const chatHistoryScreen = document.getElementById('chat-history-screen');
const chatTestScreen = document.getElementById('chat-test-screen');
const loginError = document.getElementById('login-error');
const signupError = document.getElementById('signup-error');
const toSignup = document.getElementById('to-signup');
const toLogin = document.getElementById('to-login');
const userNameEl = document.getElementById('user-name');
const userRoleEl = document.getElementById('user-role');

// Auth State Check
function checkAuthState() {
    if (authToken) {
        showApp();
        // Set the username in the sidebar
        if (localStorage.getItem('userName')) {
            userNameEl.textContent = localStorage.getItem('userName');
            userRoleEl.textContent = localStorage.getItem('userRole') || 'Administrator';
        }
        return true;
    } else {
        showAuth();
        return false;
    }
}

// Show auth screens
function showAuth() {
    if (authContainer) authContainer.style.display = 'flex';
    if (appContainer) appContainer.style.display = 'none';
}

// Show app screens
function showApp() {
    if (authContainer) authContainer.style.display = 'none';
    if (appContainer) appContainer.style.display = 'flex';
    navigate('dashboard');
}

// Switch between login and signup
if (toSignup) {
    toSignup.addEventListener('click', (e) => {
        e.preventDefault();
        loginScreen.style.display = 'none';
        signupScreen.style.display = 'block';
    });
}

if (toLogin) {
    toLogin.addEventListener('click', (e) => {
        e.preventDefault();
        signupScreen.style.display = 'none';
        loginScreen.style.display = 'block';
    });
}

// Handle login form submission
if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;
        
        // For demo purposes, hardcoded credentials
        if (username === 'admin' && password === 'password') {
            authToken = 'admin:password';
            localStorage.setItem('authToken', authToken);
            localStorage.setItem('userName', username);
            localStorage.setItem('userRole', 'Administrator');
            loginError.textContent = '';
            
            // Animation for successful login
            loginForm.classList.add('success');
            setTimeout(() => {
                showApp();
                loginForm.classList.remove('success');
            }, 800);
        } else {
            loginError.textContent = 'Invalid username or password';
            loginForm.classList.add('error');
            setTimeout(() => {
                loginForm.classList.remove('error');
            }, 500);
        }
    });
}

// Handle signup form submission
if (signupForm) {
    signupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const fullname = document.getElementById('signup-fullname').value;
        const email = document.getElementById('signup-email').value;
        const username = document.getElementById('signup-username').value;
        const password = document.getElementById('signup-password').value;
        
        // For demo purposes, always succeed
        authToken = 'user:' + username;
        localStorage.setItem('authToken', authToken);
        localStorage.setItem('userName', fullname);
        localStorage.setItem('userRole', 'User');
        signupError.textContent = '';
        
        // Animation for successful signup
        signupForm.classList.add('success');
        setTimeout(() => {
            showApp();
            signupForm.classList.remove('success');
        }, 800);
    });
}

// Handle logout
const logoutBtn = document.getElementById('logout-btn');
if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
        localStorage.removeItem('authToken');
        localStorage.removeItem('userName');
        localStorage.removeItem('userRole');
        authToken = null;
        showAuth();
    });
}

// Navigation function
function navigate(page) {
    // Hide all screens
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.add('hidden');
    });
    
    // Remove active class from all nav items
    document.querySelectorAll('.sidebar-nav li').forEach(item => {
        item.classList.remove('active');
    });
    
    // Show selected screen and mark nav item as active
    document.querySelector(`#${page}-screen`).classList.remove('hidden');
    document.querySelector(`.sidebar-nav li[data-page="${page}"]`).classList.add('active');
    
    // Load page-specific data
    if (page === 'dashboard') {
        loadDashboardData();
    } else if (page === 'agents') {
        loadAgents();
    } else if (page === 'knowledge') {
        loadKnowledgeBases();
    } else if (page === 'chat-test') {
        loadTestAgents();
    }
}

// Improved API Request with better error handling
async function apiRequest(endpoint, method = 'GET', body = null) {
    try {
        const headers = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`
        };
        
        const options = {
            method: method,
            headers: headers
        };
        
        if (body) {
            options.body = JSON.stringify(body);
        }
        
        console.log(`📡 Making ${method} request to ${API_URL}${endpoint}`);
        const response = await fetch(`${API_URL}${endpoint}`, options);
        
        if (!response.ok) {
            console.error(`❌ API error: ${response.status} ${response.statusText}`);
            
            if (response.status === 401) {
                console.log('🔄 Using dummy data due to auth error');
                
                // Return dummy data based on endpoint
                if (endpoint === '/agents') return [];
                if (endpoint === '/knowledge_bases') return [];
                return {};
            }
            
            if (method === 'POST' && endpoint === '/chat') {
                return { 
                    id: 'fallback-' + Date.now(),
                    response: "The server couldn't process your request. The AI service might be unavailable. Please check your configuration."
                };
            }
            
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log(`✅ API response received for ${endpoint}`);
        return data;
    } catch (error) {
        console.error('❌ API Request error:', error);
        
        // Add context-specific fallbacks
        if (endpoint.includes('/chat')) {
            return { 
                id: 'error-' + Date.now(),
                response: "Sorry, I encountered an error processing your request. The server might be unavailable."
            };
        }
        
        // Return empty data rather than failing
        return endpoint.includes('agents') || endpoint.includes('knowledge') ? [] : {};
    }
}

// Dashboard data loading
async function loadDashboardData() {
    try {
        // Load counts
        const agents = await apiRequest('/agents');
        const kbs = await apiRequest('/knowledge_bases');
        
        // Update dashboard cards
        document.getElementById('agent-count').textContent = agents.length || 0;
        document.getElementById('kb-count').textContent = kbs.length || 0;
        document.getElementById('conversation-count').textContent = '3'; // Default sample data
        
        // Sample conversation data
        const conversations = [
            { user_id: 'user123', agent: 'Pharmacy Agent', channel: 'Telegram', message: 'How do I take this medication?', time: '2023-06-15 14:30' },
            { user_id: 'user456', agent: 'Infusion Agent', channel: 'Web', message: 'What are the side effects?', time: '2023-06-15 13:45' },
            { user_id: 'user789', agent: 'General CS', channel: 'Telegram', message: 'When does your store open?', time: '2023-06-15 12:15' }
        ];
        
        // Update table
        const tableBody = document.querySelector('#recent-conversations tbody');
        tableBody.innerHTML = '';
        
        conversations.forEach(convo => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${convo.user_id}</td>
                <td>${convo.agent}</td>
                <td>${convo.channel}</td>
                <td>${convo.message}</td>
                <td>${convo.time}</td>
            `;
            tableBody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

// Agents screen functions
async function loadAgents() {
    try {
        const agents = await apiRequest('/agents');
        const tableBody = document.querySelector('#agents-table tbody');
        tableBody.innerHTML = '';
        
        if (!agents.length) {
            tableBody.innerHTML = '<tr><td colspan="5" class="text-center">No agents found</td></tr>';
            return;
        }
        
        agents.forEach(agent => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${agent.name}</td>
                <td>${agent.type}</td>
                <td>${agent.knowledge_base_id || 'None'}</td>
                <td>${new Date(agent.created_at).toLocaleString()}</td>
                <td>
                    <button class="btn btn-danger btn-small agent-delete" data-id="${agent.id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            `;
            tableBody.appendChild(row);
        });
        
        // Add delete event listeners
        document.querySelectorAll('.agent-delete').forEach(btn => {
            btn.addEventListener('click', async () => {
                if (confirm('Are you sure you want to delete this agent?')) {
                    try {
                        await apiRequest(`/agents/${btn.getAttribute('data-id')}`, 'DELETE');
                        loadAgents();
                    } catch (error) {
                        console.error('Error deleting agent:', error);
                    }
                }
            });
        });
    } catch (error) {
        console.error('Error loading agents:', error);
    }
}

async function handleAgentSubmit(e) {
    e.preventDefault();
    
    const agent = {
        name: document.getElementById('agent-name').value,
        type: document.getElementById('agent-type').value,
        system_prompt: document.getElementById('agent-system-prompt').value,
        moderation_prompt: document.getElementById('agent-moderation-prompt').value || null
    };
    
    const kbId = document.getElementById('agent-kb').value;
    if (kbId) {
        agent.knowledge_base_id = kbId;
    }
    
    try {
        await apiRequest('/agents', 'POST', agent);
        document.getElementById('agent-modal').style.display = 'none';
        document.getElementById('agent-form').reset();
        loadAgents();
    } catch (error) {
        console.error('Error creating agent:', error);
        alert('Error creating agent: ' + error.message);
    }
}

// Knowledge Base functions
async function loadKnowledgeBases(selectId = null) {
    try {
        const kbs = await apiRequest('/knowledge_bases');
        
        if (selectId) {
            // Populate select dropdown
            const select = document.getElementById(selectId);
            const currentValue = select.value;
            select.innerHTML = '<option value="">None</option>';
            
            kbs.forEach(kb => {
                const option = document.createElement('option');
                option.value = kb.id;
                option.textContent = kb.name;
                select.appendChild(option);
            });
            
            if (currentValue) {
                select.value = currentValue;
            }
        } else {
            // Update table
            const tableBody = document.querySelector('#kb-table tbody');
            tableBody.innerHTML = '';
            
            if (!kbs.length) {
                tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No knowledge bases found</td></tr>';
                return;
            }
            
            kbs.forEach(kb => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${kb.name}</td>
                    <td>${kb.content_preview || 'No preview available'}</td>
                    <td>${new Date(kb.created_at).toLocaleString()}</td>
                    <td>
                        <button class="btn btn-danger btn-small kb-delete" data-id="${kb.id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                `;
                tableBody.appendChild(row);
            });
            
            // Add delete event listeners
            document.querySelectorAll('.kb-delete').forEach(btn => {
                btn.addEventListener('click', async () => {
                    if (confirm('Are you sure you want to delete this knowledge base?')) {
                        try {
                            await apiRequest(`/knowledge_bases/${btn.getAttribute('data-id')}`, 'DELETE');
                            loadKnowledgeBases();
                        } catch (error) {
                            console.error('Error deleting knowledge base:', error);
                        }
                    }
                });
            });
        }
    } catch (error) {
        console.error('Error loading knowledge bases:', error);
    }
}

async function handleKBSubmit(e) {
    e.preventDefault();
    
    const kb = {
        name: document.getElementById('kb-name').value,
        content: document.getElementById('kb-content').value
    };
    
    try {
        await apiRequest('/knowledge_bases', 'POST', kb);
        document.getElementById('kb-modal').style.display = 'none';
        document.getElementById('kb-form').reset();
        loadKnowledgeBases();
    } catch (error) {
        console.error('Error creating knowledge base:', error);
        alert('Error creating knowledge base: ' + error.message);
    }
}

// Chat History functions
async function fetchChatHistory() {
    const userId = document.getElementById('user-id-filter').value;
    
    if (!userId) {
        alert('Please enter a user ID');
        return;
    }
    
    try {
        const history = await apiRequest(`/chat/history/${userId}`);
        const container = document.querySelector('.chat-history-container');
        
        if (!history || history.length === 0) {
            container.innerHTML = '<p>No chat history found for this user.</p>';
            return;
        }
        
        container.innerHTML = '';
        
        history.forEach(entry => {
            const chatEntry = document.createElement('div');
            chatEntry.className = 'chat-entry';
            chatEntry.innerHTML = `
                <div class="chat-meta">
                    <span>Agent: ${entry.agent_id}</span>
                    <span>Channel: ${entry.channel}</span>
                    <span>Time: ${new Date(entry.created_at).toLocaleString()}</span>
                </div>
                <div class="chat-message">
                    <strong>User:</strong> ${entry.message}
                </div>
                <div class="chat-response">
                    <strong>Agent:</strong> ${entry.response}
                </div>
            `;
            container.appendChild(chatEntry);
        });
    } catch (error) {
        console.error('Error fetching chat history:', error);
        document.querySelector('.chat-history-container').innerHTML = 
            '<p>Error fetching chat history. Please try again.</p>';
    }
}

// Test Chat functions
async function loadTestAgents() {
    try {
        const agents = await apiRequest('/agents');
        const select = document.getElementById('test-agent');
        select.innerHTML = '';
        
        if (!agents.length) {
            select.innerHTML = '<option value="">No agents available</option>';
            return;
        }
        
        agents.forEach(agent => {
            const option = document.createElement('option');
            option.value = agent.id;
            option.textContent = `${agent.name} (${agent.type})`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading test agents:', error);
    }
}

async function sendTestMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    const agentId = document.getElementById('test-agent').value;
    const userId = document.getElementById('test-user-id').value;
    
    if (!agentId) {
        alert('Please select an agent');
        return;
    }
    
    // Add user message to chat
    const chatMessages = document.getElementById('test-messages');
    const userMessageEl = document.createElement('div');
    userMessageEl.className = 'chat-message-bubble user-message';
    userMessageEl.textContent = message;
    chatMessages.appendChild(userMessageEl);
    
    // Add loading indicator
    const loadingEl = document.createElement('div');
    loadingEl.className = 'loading';
    loadingEl.innerHTML = '<div></div><div></div><div></div>';
    chatMessages.appendChild(loadingEl);
    
    // Clear input
    input.value = '';
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    try {
        // Send message to API
        const response = await apiRequest('/chat', 'POST', {
            user_id: userId,
            agent_id: agentId,
            message: message,
            channel: 'web'
        });
        
        // Remove loading indicator
        loadingEl.remove();
        
        // Add agent response
        const agentMessageEl = document.createElement('div');
        agentMessageEl.className = 'chat-message-bubble agent-message';
        agentMessageEl.textContent = response.response || "Sorry, I couldn't process your request.";
        chatMessages.appendChild(agentMessageEl);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        // Remove loading indicator
        loadingEl.remove();
        
        // Add error message
        const errorMessageEl = document.createElement('div');
        errorMessageEl.className = 'system-message';
        errorMessageEl.textContent = 'Error sending message. Please try again.';
        chatMessages.appendChild(errorMessageEl);
        
        console.error('Error sending test message:', error);
    }
}

// Add after DOM content loaded
function addHelpTooltips() {
    // Add tooltips to sections that might need explanation
    const helpTips = [
        { selector: '#add-agent-btn', text: 'Create an AI agent to respond to customer queries' },
        { selector: '#add-kb-btn', text: 'Add knowledge that agents can use to answer questions' },
        { selector: '[data-page="chat-test"]', text: 'Test how agents respond to different questions' },
        { selector: '#test-agent', text: 'Select which agent to talk with' }
    ];
    
    helpTips.forEach(tip => {
        const el = document.querySelector(tip.selector);
        if (el) {
            el.title = tip.text;
            // Optional: Add a help icon next to important elements
            if (['#add-agent-btn', '#add-kb-btn'].includes(tip.selector)) {
                el.innerHTML += ' <i class="fas fa-question-circle" style="font-size: 0.8em;"></i>';
            }
        }
    });
}

// Add system status check
async function checkSystemStatus() {
    try {
        // Perform a simple GET request to check API availability
        const response = await fetch(`${API_URL}/agents`);
        
        if (!response.ok) {
            showStatusBanner("⚠️ Backend API unavailable. Some features may not work properly.");
            return false;
        }
        
        return true;
    } catch (error) {
        showStatusBanner("⚠️ Cannot connect to backend server. Please check if the server is running.");
        return false;
    }
}

function showStatusBanner(message) {
    // Create a banner if it doesn't exist
    let banner = document.getElementById('status-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'status-banner';
        banner.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#f8d7da;color:#721c24;padding:10px;text-align:center;z-index:9999;';
        document.body.prepend(banner);
    }
    
    banner.textContent = message;
}

// Add guidance for new users
function showGuidance() {
    const hasSeenGuidance = localStorage.getItem('seenGuidance');
    
    if (!hasSeenGuidance) {
        const guidanceHtml = `
            <div style="padding:20px;background:#e3f2fd;border-radius:5px;margin-bottom:20px;box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                <h3>🚀 Getting Started</h3>
                <ol>
                    <li><b>Create an Agent</b>: Go to the Agents tab and add a new chatbot agent</li>
                    <li><b>Add Knowledge</b>: Create a knowledge base with information your agent can use</li>
                    <li><b>Test Your Agent</b>: Go to Test Chat to try out your new agent</li>
                </ol>
                <p>Note: If agents don't respond, check your Google API key configuration.</p>
                <button id="dismiss-guidance" class="btn btn-primary">Got it!</button>
            </div>
        `;
        
        const dashboardContent = document.querySelector('#dashboard-screen h1');
        if (dashboardContent) {
            const guidanceEl = document.createElement('div');
            guidanceEl.innerHTML = guidanceHtml;
            dashboardContent.after(guidanceEl);
            
            document.getElementById('dismiss-guidance').addEventListener('click', () => {
                guidanceEl.remove();
                localStorage.setItem('seenGuidance', 'true');
            });
        }
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Application starting...');
    
    // Check authentication state
    if (checkAuthState()) {
        navigate('dashboard');
    }
    
    // Setup navigation
    document.querySelectorAll('.sidebar-nav li').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.getAttribute('data-page');
            navigate(page);
        });
    });
    
    // Show app container and sidebar
    const sidebar = document.querySelector('.app-container > .sidebar');
    if (sidebar) {
        sidebar.style.display = 'flex';
        sidebar.classList.remove('hidden');
    }
    
    // Check system status
    checkSystemStatus();
    
    // Show guidance for new users
    showGuidance();
    
    // Agent form
    const addAgentBtn = document.getElementById('add-agent-btn');
    if (addAgentBtn) {
        addAgentBtn.addEventListener('click', () => {
            document.getElementById('agent-modal').style.display = 'block';
            loadKnowledgeBases('agent-kb');
        });
    }
    
    const agentForm = document.getElementById('agent-form');
    if (agentForm) {
        agentForm.addEventListener('submit', handleAgentSubmit);
    }
    
    // Knowledge Base form
    const addKbBtn = document.getElementById('add-kb-btn');
    if (addKbBtn) {
        addKbBtn.addEventListener('click', () => {
            document.getElementById('kb-modal').style.display = 'block';
        });
    }
    
    const kbForm = document.getElementById('kb-form');
    if (kbForm) {
        kbForm.addEventListener('submit', handleKBSubmit);
    }
    
    // Chat history
    const fetchHistoryBtn = document.getElementById('fetch-history-btn');
    if (fetchHistoryBtn) {
        fetchHistoryBtn.addEventListener('click', fetchChatHistory);
    }
    
    // Test chat
    const sendMessageBtn = document.getElementById('send-message');
    if (sendMessageBtn) {
        sendMessageBtn.addEventListener('click', sendTestMessage);
    }
    
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendTestMessage();
            }
        });
    }
    
    // Close modals
    document.querySelectorAll('.close-modal').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.modal').forEach(modal => {
                modal.style.display = 'none';
            });
        });
    });
    
    // Add help tooltips
    addHelpTooltips();
});