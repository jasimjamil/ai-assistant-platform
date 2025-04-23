import os
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import google.generativeai as genai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
import telebot

from telebot.types import Message
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import sqlite3
import time
from functools import wraps
from collections import defaultdict
from dotenv import load_dotenv
import logging
import random

# Load environment variables from .env file
load_dotenv()

# Initialize environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-key-here")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7863013922:AAFP3vi2EP9eVoVQpahE6Nfnqri4-x9zCWg")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")

# Fix Google API configuration - improved error handling and model specification
MODEL_ID = "gemini-pro"  # Changed to "gemini-pro" which is more widely available

# Print API key length for debugging (avoid printing the full key for security)
api_key_length = len(GOOGLE_API_KEY) if GOOGLE_API_KEY else 0
print(f"API Key loaded, length: {api_key_length} chars")

# Initialize Google API with proper error handling
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Check which version/style of the library we have
    print("Testing Google AI library compatibility...")
    
    # Try different methods based on what's available
    if hasattr(genai, 'GenerativeModel'):
        # Newer version
        print("Using newer Google AI library with GenerativeModel")
        model = genai.GenerativeModel(MODEL_ID)
        test_response = model.generate_content("Hello, this is a test.")
        print("✅ Google API key is valid and working with newer library")
    elif hasattr(genai, 'generate_text'):
        # Older version
        print("Using older Google AI library with generate_text")
        model = "older_version"  # Just a marker
        test_response = genai.generate_text(model=MODEL_ID, prompt="Hello, this is a test.")
        print("✅ Google API key is valid and working with older library")
    else:
        print("❌ Unrecognized Google AI library version")
        model = None
except Exception as e:
    print(f"❌ Google API initialization error: {e}")
    print("⚠️ Will use fallback responses for all queries")
    model = None

# Fix: Set a flag to run in "web-only" mode when Telegram is unavailable
TELEGRAM_AVAILABLE = False

# Fix: Initialize telegram with proxy option for networks with connectivity issues
def initialize_telegram_bot(max_retries=3, base_delay=5):
    global TELEGRAM_AVAILABLE
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries} to connect to Telegram")
            # Try with HTTP proxy instead of SOCKS
            if attempt >= 1:
                print("Attempting with HTTP proxy...")
                telebot.apihelper.proxy = {'https': 'http://127.0.0.1:8080'}  # Standard HTTP proxy
                # You can also try without any proxy:
                if attempt >= 2:
                    print("Attempting direct connection with increased timeout...")
                    telebot.apihelper.proxy = None
                    telebot.apihelper.CONNECT_TIMEOUT = 30
            
            bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=False)
            bot_info = bot.get_me()
            print(f"✅ Telegram bot connected as @{bot_info.username}")
            TELEGRAM_AVAILABLE = True
            return bot
        except Exception as e:
            print(f"⚠️ Telegram connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"❌ All {max_retries} attempts to connect to Telegram failed.")
                print("⚠️ Running in web-only mode, Telegram functionality will be disabled")
                TELEGRAM_AVAILABLE = False
                return None

# Initialize the bot with retry logic
bot = initialize_telegram_bot()

# Create FastAPI app - no lifespan at creation time
app = FastAPI(title="Agentic CS System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instead, mount static files at /static (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize SQLite database
conn = sqlite3.connect("agentic_cs.db", check_same_thread=False)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    knowledge_base_id TEXT,
    moderation_prompt TEXT,
    created_at TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    channel TEXT NOT NULL,
    created_at TEXT NOT NULL
)
''')

conn.commit()

# Function to run Telegram bot in a separate thread
def run_telegram_bot():
    if bot:
        try:
            print("Starting Telegram bot polling...")
            bot.infinity_polling()
        except Exception as e:
            print(f"❌ Telegram polling error: {e}")
    else:
        print("⚠️ Telegram bot not initialized. Polling not started.")

@app.on_event("startup")
async def startup_event():
    print("Starting up the server...")
    print(f"Telegram bot available: {TELEGRAM_AVAILABLE}")
    print(f"Google API available: {model is not None}")
    
    # We'll handle static files directly instead of mounting
    print("Server is ready to handle requests!")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down the server...")

# Security
security = HTTPBearer()

# Pydantic models
class Agent(BaseModel):
    id: Optional[str] = None
    name: str
    type: str
    system_prompt: str
    knowledge_base_id: Optional[str] = None
    moderation_prompt: Optional[str] = None

class KnowledgeBase(BaseModel):
    id: Optional[str] = None
    name: str
    content: str

class ChatMessage(BaseModel):
    user_id: str
    agent_id: str
    message: str
    channel: str = "web"

class ChatResponse(BaseModel):
    id: str
    response: str

class UserAuth(BaseModel):
    username: str
    password: str

# Vector store cache
vector_stores = {}

# Helper functions
def get_current_time():
    return datetime.now().isoformat()

def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Simple token validation
    token = credentials.credentials
    if token != f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return True

# Add a retry decorator for API calls
def retry_on_error(max_retries=3, initial_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if retry == max_retries - 1:
                        print(f"❌ Failed after {max_retries} retries: {e}")
                        raise
                    print(f"⚠️ Attempt {retry + 1} failed: {e}, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        return wrapper
    return decorator

# Create a dedicated fallback AI system that works without any external API
class FallbackAI:
    """A simple fallback AI that works even when all APIs are down."""
    
    def __init__(self):
        self.responses = {
            "greeting": [
                "Hello! I'm your AI assistant. How can I help you today?",
                "Hi there! I'm currently running in local mode. What can I do for you?",
                "Greetings! I'm here to assist you, though I'm running with limited capabilities."
            ],
            "farewell": [
                "Goodbye! Feel free to return if you have more questions.",
                "Take care! I'll be here if you need assistance later.",
                "Until next time! Have a great day."
            ],
            "gratitude": [
                "You're welcome! Is there anything else I can help with?",
                "Happy to help! Let me know if you need anything else.",
                "My pleasure! What else would you like to know?"
            ],
            "default": [
                "I understand you're asking about '{topic}'. While I'm operating in offline mode, I can tell you that this is an interesting topic.",
                "You're inquiring about '{topic}'. I'm currently running with limited capabilities, but I'd be happy to discuss this when my full systems are online.",
                "'{topic}' is what you're interested in. I wish I could provide more detailed information, but I'm currently operating with restricted access to my knowledge base.",
                "I notice you're asking about '{topic}'. I'm running in local mode right now, but I can try to assist with basic information.",
                "Regarding '{topic}', I'm currently operating with limited functionality, but I can still try to help with simple questions."
            ],
            "error": [
                "I apologize, but I'm experiencing technical difficulties at the moment. My team is working to restore full functionality.",
                "I'm currently running in emergency backup mode due to connectivity issues. I have limited capabilities right now.",
                "I'm sorry, but my knowledge retrieval system is temporarily offline. I can only provide basic responses at this time.",
                "My advanced AI functions are currently unavailable. I'm operating with a simplified response system."
            ]
        }
        logger.info("✅ Fallback AI system initialized successfully")
    
    def extract_topic(self, text):
        """Extract a simple topic from the input text."""
        # Remove common question words and punctuation
        text = text.lower()
        for word in ["what", "why", "how", "when", "where", "who", "is", "are", "can", "could", "would", "?", "."]:
            text = text.replace(word, "")
        
        # Find the longest word as the topic (simple approach)
        words = text.split()
        if not words:
            return "that"
        
        topic = max(words, key=len)
        return topic if len(topic) > 3 else "that topic"
    
    def get_response_type(self, text):
        """Determine the type of response needed."""
        text = text.lower()
        
        # Check for greetings
        if any(greeting in text for greeting in ["hello", "hi ", "hey", "greetings"]):
            return "greeting"
            
        # Check for farewells
        if any(farewell in text for farewell in ["bye", "goodbye", "see you", "farewell"]):
            return "farewell"
            
        # Check for gratitude
        if any(thanks in text for thanks in ["thank", "thanks", "appreciate"]):
            return "gratitude"
            
        # Default to topic-based response
        return "default"
    
    def generate_response(self, prompt):
        """Generate a fallback response."""
        try:
            response_type = self.get_response_type(prompt)
            
            if response_type == "default":
                topic = self.extract_topic(prompt)
                template = random.choice(self.responses[response_type])
                return template.format(topic=topic)
            else:
                return random.choice(self.responses[response_type])
        except Exception as e:
            logger.error(f"Error in fallback response: {str(e)}")
            return random.choice(self.responses["error"])

# Simplify Google AI initialization to handle API key issues better
def initialize_google_ai():
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            logger.error("❌ Google API key not found. Set GOOGLE_API_KEY environment variable.")
            return None
        
        # Log only the first few characters of the API key for debugging (never log full keys)
        key_prefix = GOOGLE_API_KEY[:4] + "..." if len(GOOGLE_API_KEY) > 4 else "too_short"
        logger.info(f"Initializing Google AI with API key prefix: {key_prefix}")
        
        # Try the newer library first (gemini models)
        try:
            import google.generativeai as genai
            
            # Configure with API key
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Get available models to verify API key works
            try:
                models = genai.list_models()
                model_names = [model.name for model in models]
                logger.info(f"Available models: {model_names}")
                
                # Check if gemini-pro is in the models list
                if any("gemini-pro" in model_name for model_name in model_names):
                    # Use gemini-pro model
                    model = genai.GenerativeModel("gemini-pro")
                    
                    # Verify with a simple test
                    response = model.generate_content("Test connection")
                    if hasattr(response, 'text') and response.text:
                        logger.info("✅ Google AI initialized successfully with gemini-pro")
                        return {"client": model, "type": "gemini"}
                else:
                    logger.warning("⚠️ Gemini models not available with this API key")
            except Exception as e:
                logger.error(f"❌ Error listing models: {str(e)}")
        except ImportError:
            logger.warning("google.generativeai library not available")
        except Exception as e:
            logger.error(f"❌ Error with newer Google AI library: {str(e)}")
        
        # Try older PaLM API as fallback
        try:
            from google.ai import generativelanguage as glm
            from google.api_core.client_options import ClientOptions
            
            # Try with text models
            client_options = ClientOptions(api_key=GOOGLE_API_KEY)
            model_client = glm.TextServiceClient(client_options=client_options)
            
            # Test with a simple prompt - use correct model format
            try:
                request = glm.GenerateTextRequest(
                    model="models/text-bison-001",
                    prompt=glm.TextPrompt(text="Test connection"),
                )
                response = model_client.generate_text(request)
                
                if response and hasattr(response, 'candidates') and len(response.candidates) > 0:
                    logger.info("✅ Google AI initialized successfully with text-bison-001")
                    return {"client": model_client, "type": "palm-text"}
            except Exception as e:
                logger.error(f"❌ Error testing text model: {str(e)}")
            
            # Try with chat models
            try:
                chat_client = glm.DiscussServiceClient(client_options=client_options)
                request = glm.GenerateMessageRequest(
                    model="models/chat-bison-001",
                    prompt=glm.MessagePrompt(messages=[glm.Message(content="Test connection")])
                )
                response = chat_client.generate_message(request)
                
                if response and hasattr(response, 'candidates') and len(response.candidates) > 0:
                    logger.info("✅ Google AI initialized successfully with chat-bison-001")
                    return {"client": chat_client, "type": "palm-chat"}
            except Exception as e:
                logger.error(f"❌ Error testing chat model: {str(e)}")
                
        except ImportError:
            logger.warning("google.ai.generativelanguage library not available")
        except Exception as e:
            logger.error(f"❌ Error with older Google AI library: {str(e)}")
        
        # If we get here, all attempts failed
        logger.error("❌ All Google AI initialization attempts failed")
        return None
        
    except Exception as e:
        logger.error(f"❌ Unexpected error initializing Google AI: {str(e)}")
        return None

# Completely revised generate_response function with robust fallback
async def generate_response(prompt, history=None):
    model_data = get_ai_model()
    fallback = get_fallback_ai()
    
    # If no AI model is available, use fallback immediately
    if model_data is None:
        logger.warning("No AI model available, using fallback")
        return fallback.generate_response(prompt)
    
    # Try to use the AI model with timeout protection
    try:
        model_type = model_data.get("type", "unknown")
        client = model_data.get("client")
        
        logger.info(f"Attempting to generate response using model type: {model_type}")
        
        # Different handling based on model type
        if model_type == "gemini":
            # Using gemini model
            try:
                # Set a timeout for the API call
                timeout_seconds = 10
                
                # Simulate async timeout since the generativeai library doesn't support true async
                start_time = time.time()
                
                # Process history if provided
                if history and len(history) > 0:
                    chat = client.start_chat(history=history)
                    response = chat.send_message(prompt)
                else:
                    response = client.generate_content(prompt)
                
                # Check if it took too long
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Response generation took too long: {time.time() - start_time:.2f}s")
                    raise TimeoutError("API call timed out")
                
                # Extract text from the response
                if hasattr(response, 'text'):
                    return response.text
                else:
                    return str(response)
            except Exception as e:
                logger.error(f"Error with gemini model: {str(e)}")
                return fallback.generate_response(prompt)
                
        elif model_type == "palm-text":
            # Using PaLM text model
            try:
                from google.ai import generativelanguage as glm
                
                request = glm.GenerateTextRequest(
                    model="models/text-bison-001",
                    prompt=glm.TextPrompt(text=prompt),
                    temperature=0.7,
                    max_output_tokens=1024
                )
                
                # Generate response
                response = client.generate_text(request)
                
                # Check response
                if hasattr(response, 'candidates') and len(response.candidates) > 0:
                    return response.candidates[0].output
                else:
                    return fallback.generate_response(prompt)
            except Exception as e:
                logger.error(f"Error with PaLM text model: {str(e)}")
                return fallback.generate_response(prompt)
                
        elif model_type == "palm-chat":
            # Using PaLM chat model
            try:
                from google.ai import generativelanguage as glm
                
                # Process history
                messages = []
                if history and len(history) > 0:
                    for msg in history:
                        author = msg.get("role", "user")
                        content = msg.get("content", "")
                        messages.append(glm.Message(author=author, content=content))
                
                # Add current prompt
                messages.append(glm.Message(content=prompt))
                
                # Create request
                request = glm.GenerateMessageRequest(
                    model="models/chat-bison-001",
                    prompt=glm.MessagePrompt(messages=messages)
                )
                
                # Generate response
                response = client.generate_message(request)
                
                # Check response
                if hasattr(response, 'candidates') and len(response.candidates) > 0:
                    return response.candidates[0].content
                else:
                    return fallback.generate_response(prompt)
            except Exception as e:
                logger.error(f"Error with PaLM chat model: {str(e)}")
                return fallback.generate_response(prompt)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return fallback.generate_response(prompt)
            
    except Exception as e:
        logger.error(f"Unexpected error generating response: {str(e)}")
        return fallback.generate_response(prompt)

# Global variables for models
ai_model = None
fallback_ai = None

# Getter functions
def get_ai_model():
    global ai_model
    return ai_model
    
def get_fallback_ai():
    global fallback_ai
    return fallback_ai

# Update your startup function
@app.on_event("startup")
async def startup_event():
    global ai_model, fallback_ai
    
    # Always initialize the fallback AI first
    logger.info("Initializing Fallback AI system...")
    fallback_ai = FallbackAI()
    
    # Try to initialize Google AI
    logger.info("Initializing Google AI...")
    ai_model = initialize_google_ai()
    
    if ai_model:
        logger.info(f"✅ Google AI initialized successfully with model type: {ai_model.get('type')}")
    else:
        logger.warning("⚠️ Google AI initialization failed, will use Fallback AI system")
    
    # ... rest of your startup code ...

# Make sure the API endpoints use the new generate_response function
@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "")
        history = data.get("history", [])
        
        if not message:
            return JSONResponse(status_code=400, content={"error": "No message provided"})
        
        # Get AI response
        ai_response = await generate_response(message, history)
        
        return {"response": ai_response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": "Server error", "message": str(e)}
        )

# Replace existing model initialization with this

# Fix authentication middleware
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Public paths that don't need authentication
    public_paths = ["/", "/login", "/signup", "/api/auth/login", "/api/auth/signup", 
                   "/styles.css", "/app.js", "/script.js", "/test", "/test-page"]
    
    # Skip auth for static files
    path = request.url.path
    if path.startswith("/static") or any(path.endswith(ext) for ext in [".css", ".js", ".html", ".png", ".jpg", ".svg"]):
        return await call_next(request)
        
    # Skip auth for public paths
    if any(path == public_path for public_path in public_paths):
        return await call_next(request)
    
    # Simple auth check - a token in headers or cookies
    auth_token = request.headers.get("Authorization") or request.cookies.get("auth_token")
    
    # For demo purposes - accept any token for now
    if auth_token:
        return await call_next(request)
    else:
        # Return a friendlier JSON error instead of plain 401
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required", "message": "Please log in to access this resource"}
        )

# Add proper auth endpoints
@app.post("/api/auth/login")
async def login(request: Request):
    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        
        # Simplified auth for demo - in production use proper password checking
        if username and password:
            # Generate a simple token
            token = f"user-token-{username}-{int(time.time())}"
            return {"success": True, "token": token, "username": username}
        else:
            return JSONResponse(status_code=400, content={"error": "Missing username or password"})
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Server error during login"})

@app.post("/api/auth/signup")
async def signup(request: Request):
    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        email = data.get("email")
        
        # Simplified signup for demo
        if username and password:
            # Generate a simple token
            token = f"user-token-{username}-{int(time.time())}"
            return {"success": True, "token": token, "username": username}
        else:
            return JSONResponse(status_code=400, content={"error": "Missing required fields"})
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Server error during signup"})

# Fix agents API
@app.get("/api/agents")
async def get_agents():
    # Return dummy data if no real data available
    return [
        {"id": 1, "name": "General Assistant", "description": "A general-purpose AI assistant"},
        {"id": 2, "name": "Code Helper", "description": "Specialized in coding assistance"},
        {"id": 3, "name": "Research Agent", "description": "Helps with research and information gathering"}
    ]

@app.get("/api/knowledge_bases")
async def get_knowledge_bases():
    # Return dummy data if no real data available
    return [
        {"id": 1, "name": "General Knowledge", "description": "Contains general knowledge information"},
        {"id": 2, "name": "Programming", "description": "Programming languages and frameworks"},
        {"id": 3, "name": "Science", "description": "Scientific concepts and discoveries"}
    ]

# API Endpoints
@app.post("/api/agents", dependencies=[Depends(authenticate)])
async def create_agent(agent: Agent):
    agent_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?)",
        (agent_id, agent.name, agent.type, agent.system_prompt, agent.knowledge_base_id, agent.moderation_prompt, get_current_time())
    )
    conn.commit()
    return {"id": agent_id, "message": "Agent created successfully"}

@app.post("/api/knowledge_bases", dependencies=[Depends(authenticate)])
async def create_knowledge_base(kb: KnowledgeBase):
    kb_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO knowledge_bases VALUES (?, ?, ?, ?)",
        (kb_id, kb.name, kb.content, get_current_time())
    )
    conn.commit()
    return {"id": kb_id, "message": "Knowledge base created successfully"}

@app.post("/api/chat", dependencies=[Depends(authenticate)])
async def chat(message: ChatMessage):
    # Get the agent
    cursor.execute("SELECT * FROM agents WHERE id = ?", (message.agent_id,))
    agent_data = cursor.fetchone()
    
    if not agent_data:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = {
        "id": agent_data[0],
        "name": agent_data[1],
        "type": agent_data[2],
        "system_prompt": agent_data[3],
        "knowledge_base_id": agent_data[4],
        "moderation_prompt": agent_data[5]
    }
    
    try:
        # Generate response based on whether there's a knowledge base
        if agent["knowledge_base_id"]:
            ai_response = generate_response_with_knowledge(
                message.message, 
                agent["knowledge_base_id"], 
                agent["system_prompt"]
            )
        else:
            ai_response = generate_response(
                message.message,
                agent["system_prompt"]
            )
        
        # Apply moderation if needed
        if agent["moderation_prompt"]:
            ai_response, is_safe = moderate_response(ai_response, agent["moderation_prompt"])
            
            # If moderation failed, use fallback
            if not is_safe:
                ai_response = fallback_mechanism(message.message, agent["type"])
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        # Fallback mechanism
        ai_response = fallback_mechanism(message.message, agent["type"])
    
    # Save to chat history
    chat_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO chat_history VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chat_id, message.user_id, message.agent_id, message.message, ai_response, message.channel, get_current_time())
    )
    conn.commit()
    
    return {"id": chat_id, "response": ai_response}

@app.get("/api/chat/history/{user_id}", dependencies=[Depends(authenticate)])
async def get_chat_history(user_id: str):
    cursor.execute(
        "SELECT * FROM chat_history WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    history = cursor.fetchall()
    return [
        {
            "id": h[0],
            "user_id": h[1],
            "agent_id": h[2],
            "message": h[3],
            "response": h[4],
            "channel": h[5],
            "created_at": h[6]
        }
        for h in history
    ]

# Add a health check function that runs periodically
def check_health():
    health_status = {
        "telegram": False,
        "google_api": False
    }
    
    # Check Telegram
    try:
        bot_info = bot.get_me()
        health_status["telegram"] = True
        print(f"✅ Telegram connected as @{bot_info.username}")
    except Exception as e:
        print(f"❌ Telegram health check failed: {e}")
    
    # Check Google API
    try:
        test_response = model.generate_text(prompt="Hello")
        health_status["google_api"] = True
        print("✅ Google API responding")
    except Exception as e:
        print(f"❌ Google API health check failed: {e}")
    
    return health_status

# Add a rate limiter to prevent hitting API limits
user_last_request = defaultdict(float)
RATE_LIMIT_SECONDS = 1  # Minimum time between requests per user

def is_rate_limited(user_id):
    current_time = time.time()
    if current_time - user_last_request[user_id] < RATE_LIMIT_SECONDS:
        return True
    user_last_request[user_id] = current_time
    return False

# Fix the message handler to only attach if bot is available
if bot is not None:
    @bot.message_handler(func=lambda message: True)
    def handle_message(message):
        if is_rate_limited(message.from_user.id):
            bot.reply_to(message, "Please wait a moment before sending another message.")
            return
        
        # Get the first available agent
        cursor.execute("SELECT id FROM agents LIMIT 1")
        result = cursor.fetchone()
        
        if not result:
            bot.reply_to(message, "No agents are configured yet.")
            return
        
        agent_id = result[0]
        
        # Get the agent details
        cursor.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        agent_data = cursor.fetchone()
        
        if not agent_data:
            bot.reply_to(message, "Agent configuration error.")
            return
        
        agent = {
            "id": agent_data[0],
            "name": agent_data[1],
            "type": agent_data[2],
            "system_prompt": agent_data[3],
            "knowledge_base_id": agent_data[4],
            "moderation_prompt": agent_data[5]
        }
        
        try:
            # Generate response based on whether there's a knowledge base
            if agent["knowledge_base_id"]:
                ai_response = generate_response_with_knowledge(
                    message.text, 
                    agent["knowledge_base_id"], 
                    agent["system_prompt"]
                )
            else:
                ai_response = generate_response(
                    message.text,
                    agent["system_prompt"]
                )
            
            # Apply moderation if needed
            if agent["moderation_prompt"]:
                ai_response, is_safe = moderate_response(ai_response, agent["moderation_prompt"])
                
                # If moderation failed, use fallback
                if not is_safe:
                    ai_response = fallback_mechanism(message.text, agent["type"])
        
        except Exception as e:
            # Fallback mechanism
            ai_response = fallback_mechanism(message.text, agent["type"])
        
        # Save to chat history
        chat_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO chat_history VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, str(message.from_user.id), agent_id, message.text, ai_response, "telegram", get_current_time())
        )
        conn.commit()
        
        # Send response back to Telegram
        bot.reply_to(message, ai_response)
else:
    print("⚠️ Telegram bot message handler not set up as bot initialization failed")

# Fix: Modify the polling logic to properly handle None bot
def start_bot_polling():
    if bot is not None:
        try:
            print("Starting Telegram bot polling...")
            bot.polling(non_stop=True, interval=0)
        except Exception as e:
            print(f"❌ Error in bot polling: {e}")
    else:
        print("⚠️ Not starting bot polling as bot initialization failed")

# Only start polling in a separate thread if the bot was initialized
if bot is not None:
    import threading
    polling_thread = threading.Thread(target=start_bot_polling, daemon=True)
    polling_thread.start()
else:
    print("⚠️ Running in web-only mode. API endpoints will work but Telegram bot is unavailable.")

# Create a simple index.html in frontend directory if it doesn't exist
if not os.path.exists("frontend"):
    os.makedirs("frontend")
    
if not os.path.exists("frontend/index.html"):
    with open("frontend/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI Chat</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #chat-container { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        #user-input { width: 80%; padding: 8px; }
        button { padding: 8px 16px; }
    </style>
</head>
<body>
    <h1>AI Chat</h1>
    <div id="chat-container"></div>
    <input type="text" id="user-input" placeholder="Type your message here...">
    <button onclick="sendMessage()">Send</button>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.innerHTML = `<strong>${role}:</strong> ${content}`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            addMessage('User', message);
            userInput.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                addMessage('AI', data.response);
            } catch (error) {
                console.error('Error:', error);
                addMessage('System', 'Failed to get response from the server.');
            }
        }

        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
        """)

# Setup logging to debug frontend serving issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug endpoint to check if API is responding
@app.get("/api/status")
async def status():
    logger.info("Status endpoint called")
    return {"status": "ok", "message": "API is running"}

# Serve index.html at the root with logging
@app.get("/")
async def read_root():
    logger.info("Root path requested, serving index.html")
    index_path = "frontend/index.html"
    if os.path.exists(index_path):
        logger.info(f"File {index_path} exists, serving it")
        return FileResponse(index_path)
    else:
        logger.error(f"File {index_path} does not exist!")
        return JSONResponse(content={"error": "index.html not found"}, status_code=404)

# Serve frontend files with detailed logging
@app.get("/{path:path}")
async def read_file(path: str):
    logger.info(f"Path requested: {path}")
    
    # Special handling for common frontend files
    if path in ["app.js", "script.js", "styles.css"]:
        file_path = f"frontend/{path}"
        if os.path.isfile(file_path):
            logger.info(f"File {file_path} exists, serving it")
            return FileResponse(file_path)
        else:
            logger.error(f"File {file_path} not found!")
    
    # General file handling
    file_path = f"frontend/{path}"
    if os.path.isfile(file_path):
        logger.info(f"File {file_path} exists, serving it")
        return FileResponse(file_path)
    else:
        # If path doesn't exist, serve index.html for SPA routing
        logger.info(f"File {file_path} not found, serving index.html instead (SPA routing)")
        return FileResponse("frontend/index.html")

# General static file handler
@app.get("/{file_path:path}")
async def read_static(file_path: str):
    # First try frontend directory
    frontend_path = f"frontend/{file_path}"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    
    # Then try static directory
    static_path = f"static/{file_path}"
    if os.path.exists(static_path):
        return FileResponse(static_path)
        
    # If file doesn't exist, return 404
    raise HTTPException(status_code=404, detail="File not found")

# Simple endpoint to test API directly
@app.get("/api-test", response_class=HTMLResponse)
async def api_test():
    return "API is working! Try accessing the root URL: <a href='/'>http://localhost:8000/</a>"

# Add this test endpoint
@app.get("/test")
async def test_endpoint():
    return {"message": "Server is running!"}

# Add this near the top of your file, after imports
def create_test_file():
    with open("frontend/test.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Test Page</h1>
    <p>If you can see this, your server is working correctly!</p>
</body>
</html>
        """)

# Call this function before starting the app
create_test_file()

# Add a specific route for the test file
@app.get("/test-page")
async def test_page():
    return FileResponse("frontend/test.html")

# Import necessary libraries for Telegram
import asyncio
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram.error import TelegramError, NetworkError, TimedOut

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_bot = None

# Initialize Telegram Bot with proper error handling
async def init_telegram_bot():
    global telegram_bot
    
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ Telegram bot token not found. Set TELEGRAM_BOT_TOKEN environment variable.")
        return None
    
    try:
        # Create the bot instance
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        
        # Add message handler
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Add error handler
        application.add_error_handler(error_handler)
        
        # Start the bot in a separate task so it doesn't block the main application
        asyncio.create_task(application.run_polling(allowed_updates=Update.ALL_TYPES, 
                                                   drop_pending_updates=True,
                                                   close_loop=False))
        
        # Also initialize a Bot instance for direct API access if needed
        telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        
        logger.info("✅ Telegram bot initialized successfully")
        return telegram_bot
    except Exception as e:
        logger.error(f"❌ Telegram bot initialization failed: {str(e)}")
        return None

# Telegram command handlers
async def start_command(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "👋 Welcome to the AI Assistant Bot! I'm here to help you. "
        "You can ask me questions or start a chat. Type /help to see available commands."
    )

async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "🤖 *AI Assistant Bot Help*\n\n"
        "*Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "*How to use:*\n"
        "Simply send a message and I'll respond using AI!"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

# Message handler
async def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "User"
    
    try:
        logger.info(f"Received message from {username} (ID: {user_id}): {user_message}")
        
        # Use your existing AI chat functionality
        ai_response = await generate_response(user_message)
        
        # Send typing indicator for a more natural feel
        await update.message.chat.send_action(action="typing")
        
        # Add a small delay to simulate thinking
        await asyncio.sleep(1)
        
        # Send the response
        await update.message.reply_text(ai_response)
        
    except Exception as e:
        logger.error(f"Error in message handler: {str(e)}")
        await update.message.reply_text(
            "I'm sorry, I encountered an error while processing your message. Please try again later."
        )

# Error handler for Telegram
async def error_handler(update: Update, context: CallbackContext):
    error = context.error
    
    # Handle different types of errors
    if isinstance(error, NetworkError):
        logger.error(f"Telegram NetworkError: {str(error)}")
        # Implement exponential backoff retry logic
        if hasattr(context, 'retry_count') and context.retry_count < 5:
            context.retry_count += 1
            wait_time = 2 ** context.retry_count
            logger.info(f"Retrying in {wait_time} seconds (attempt {context.retry_count}/5)")
            await asyncio.sleep(wait_time)
            # Retry the update
            if update:
                await context.bot.process_update(update)
        
    elif isinstance(error, TimedOut):
        logger.warning(f"Telegram request timed out: {str(error)}")
        # Simply log timeouts, they're common and usually resolve themselves
        
    else:
        logger.error(f"Unhandled Telegram error: {str(error)}")
        # For other errors, try to notify user if possible
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "Sorry, I encountered a technical problem. Please try again later."
            )

# Function to send messages via Telegram (can be called from elsewhere in the app)
async def send_telegram_message(chat_id, text):
    if not telegram_bot:
        logger.error("Telegram bot not initialized")
        return False
    
    try:
        await telegram_bot.send_message(chat_id=chat_id, text=text)
        return True
    except NetworkError as e:
        logger.error(f"Network error when sending Telegram message: {str(e)}")
        # Implement retry logic
        for attempt in range(3):
            try:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds (attempt {attempt+1}/3)")
                await asyncio.sleep(wait_time)
                await telegram_bot.send_message(chat_id=chat_id, text=text)
                return True
            except NetworkError:
                continue
        return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")
        return False

# Add Telegram endpoint for webhook mode (alternative to polling)
@app.post("/api/telegram-webhook")
async def telegram_webhook(request: Request):
    if not TELEGRAM_BOT_TOKEN:
        return JSONResponse(status_code=500, content={"error": "Telegram bot token not configured"})
    
    try:
        update_data = await request.json()
        update = Update.de_json(data=update_data, bot=telegram_bot)
        
        # Process the update in a background task
        asyncio.create_task(process_telegram_update(update))
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error in Telegram webhook: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

async def process_telegram_update(update: Update):
    # This would handle the update if using webhook mode
    # Implementation would depend on your telegram-python-bot version and setup
    pass

# Update your startup function to initialize Telegram
@app.on_event("startup")
async def startup_event():
    # ... your existing startup code ...
    
    # Initialize Telegram bot
    logger.info("Initializing Telegram bot...")
    telegram_bot = await init_telegram_bot()
    
    if telegram_bot:
        logger.info("✅ Telegram bot is ready")
    else:
        logger.warning("⚠️ Telegram bot initialization failed, continuing without Telegram functionality")
    
    # ... rest of your startup code ...

# Add a status endpoint for Telegram
@app.get("/api/telegram/status")
async def telegram_status():
    if telegram_bot:
        try:
            # Test the connection to Telegram
            bot_info = await telegram_bot.get_me()
            return {
                "status": "ok", 
                "connected": True,
                "bot_name": bot_info.first_name,
                "bot_username": bot_info.username
            }
        except Exception as e:
            return {"status": "error", "connected": False, "error": str(e)}
    else:
        return {"status": "not_initialized", "connected": False}

# Add this handler function that Vercel will use
def handler(request: Request):
    return app.handle_request(request)

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True) 