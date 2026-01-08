"""
Streamlit Chatbot Application.

A modern chatbot interface powered by LangGraph and Ollama.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from chatbot import create_chatbot_graph, ToolsRegistry
from chatbot.tools import tools_registry
from chatbot.graph import chat

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 2.5rem;
        background: linear-gradient(120deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Outfit', sans-serif;
        color: #94a3b8;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .stChatMessage {
        font-family: 'Outfit', sans-serif;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput > div > div > input {
        font-family: 'Outfit', sans-serif;
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
    }
    
    .sidebar .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    .tool-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #a78bfa);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        color: #4ade80;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    div[data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95);
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 LangGraph Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by LangGraph & Ollama</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Ollama URL input
    default_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_url = st.text_input(
        "Ollama URL",
        value=default_url,
        placeholder="http://localhost:11434",
        help="Enter your Ollama server URL"
    )
    
    st.divider()
    
    # Model selection - common Ollama models
    default_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    model = st.selectbox(
        "Model",
        ["llama3.2", "llama3.1", "llama3", "mistral", "mixtral", "codellama", "phi3", "gemma2", "qwen2.5"],
        index=0 if default_model == "llama3.2" else None,
        help="Select the Ollama model to use (must be pulled first)"
    )
    
    # Custom model input
    custom_model = st.text_input(
        "Or enter custom model",
        placeholder="e.g., llama3.2:70b",
        help="Enter a specific model name if not in the list"
    )
    
    # Use custom model if provided
    selected_model = custom_model if custom_model else model
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random"
    )
    
    st.divider()
    
    # Tools section
    st.markdown("### 🛠️ Tools")
    
    registered_tools = tools_registry.list_tools()
    if registered_tools:
        st.markdown('<div class="status-indicator"><span class="status-dot"></span>Tools Active</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for tool_name in registered_tools:
            st.markdown(f'<span class="tool-badge">{tool_name}</span>', unsafe_allow_html=True)
    else:
        st.info("No tools registered yet. Add tools in `chatbot/tools.py`")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.graph = None
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = None

# Initialize/update graph when settings change
def get_graph():
    return create_chatbot_graph(
        model_name=selected_model, 
        temperature=temperature,
        base_url=ollama_url
    )

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
    
    # Get or create graph
    graph = get_graph()
    
    if graph:
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    updated_messages, response = chat(
                        graph, 
                        st.session_state.messages, 
                        prompt
                    )
                    st.session_state.messages = updated_messages
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Show welcome message if no messages
if not st.session_state.messages:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem;
        color: #94a3b8;
        font-family: 'Outfit', sans-serif;
    ">
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">👋 Welcome!</p>
        <p>Make sure Ollama is running locally, then start chatting!</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #64748b;">
            Run <code style="background: rgba(124, 58, 237, 0.2); padding: 2px 8px; border-radius: 4px;">ollama serve</code> if not started
        </p>
        <p style="font-size: 0.9rem; margin-top: 1rem; color: #64748b;">
            Add custom tools in <code style="background: rgba(124, 58, 237, 0.2); padding: 2px 8px; border-radius: 4px;">chatbot/tools.py</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
