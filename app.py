"""
Streamlit Chatbot Application.

A modern chatbot interface powered by LangGraph with support for
multiple LLM providers: Ollama, Groq, and OpenAI.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from chatbot import create_chatbot_graph
from chatbot.tools import tools_registry
from chatbot.graph import chat
from chatbot.llm_provider import LLMProvider, LLMFactory

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

    .provider-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .provider-ollama { background: linear-gradient(135deg, #22c55e, #16a34a); color: white; }
    .provider-groq { background: linear-gradient(135deg, #f97316, #ea580c); color: white; }
    .provider-openai { background: linear-gradient(135deg, #10a37f, #0d8a6a); color: white; }

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
st.markdown('<p class="sub-header">Powered by LangGraph | Ollama • Groq • OpenAI</p>', unsafe_allow_html=True)

# Provider display names and colors
PROVIDER_DISPLAY = {
    "ollama": ("🦙 Ollama", "ollama"),
    "groq": ("⚡ Groq", "groq"),
    "openai": ("🤖 OpenAI", "openai"),
}

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Provider selection
    default_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    provider_options = ["ollama", "groq", "openai"]
    provider_index = provider_options.index(default_provider) if default_provider in provider_options else 0

    provider = st.selectbox(
        "LLM Provider",
        provider_options,
        index=provider_index,
        format_func=lambda x: PROVIDER_DISPLAY.get(x, (x, x))[0],
        help="Select your LLM provider"
    )

    # Show provider badge
    provider_name, provider_class = PROVIDER_DISPLAY.get(provider, (provider, ""))
    st.markdown(f'<span class="provider-badge provider-{provider_class}">{provider_name}</span>', unsafe_allow_html=True)

    st.divider()

    # Provider-specific configuration
    api_key = None
    base_url = None

    if provider == "ollama":
        # Ollama URL input
        default_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        base_url = st.text_input(
            "Ollama URL",
            value=default_url,
            placeholder="http://localhost:11434",
            help="Enter your Ollama server URL"
        )

        # Model selection for Ollama
        default_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_models = LLMFactory.get_default_models(LLMProvider.OLLAMA)
        model = st.selectbox(
            "Model",
            ollama_models,
            index=ollama_models.index(default_model) if default_model in ollama_models else 0,
            help="Select the Ollama model to use (must be pulled first)"
        )

        # Custom model input
        custom_model = st.text_input(
            "Or enter custom model",
            placeholder="e.g., llama3.2:70b",
            help="Enter a specific model name if not in the list"
        )
        selected_model = custom_model if custom_model else model

    elif provider == "groq":
        # Groq API key
        default_api_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API Key",
            value=default_api_key,
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq API key"
        )

        if not api_key:
            st.warning("⚠️ Groq API key required")

        # Model selection for Groq
        default_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        groq_models = LLMFactory.get_default_models(LLMProvider.GROQ)
        selected_model = st.selectbox(
            "Model",
            groq_models,
            index=groq_models.index(default_model) if default_model in groq_models else 0,
            help="Select the Groq model to use"
        )

    elif provider == "openai":
        # OpenAI API key
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            value=default_api_key,
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key"
        )

        if not api_key:
            st.warning("⚠️ OpenAI API key required")

        # Model selection for OpenAI
        default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_models = LLMFactory.get_default_models(LLMProvider.OPENAI)
        selected_model = st.selectbox(
            "Model",
            openai_models,
            index=openai_models.index(default_model) if default_model in openai_models else 0,
            help="Select the OpenAI model to use"
        )

        # Optional custom base URL
        custom_base_url = st.text_input(
            "Custom Base URL (optional)",
            placeholder="https://api.openai.com/v1",
            help="For Azure OpenAI or custom endpoints"
        )
        if custom_base_url:
            base_url = custom_base_url

    st.divider()

    # Temperature slider
    default_temp = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=default_temp,
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
    try:
        return create_chatbot_graph(
            provider=provider,
            model=selected_model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url
        )
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None


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
    # Check if API key is required and not provided
    if provider in ["groq", "openai"] and not api_key:
        st.error(f"Please enter your {provider.upper()} API key in the sidebar.")
    else:
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
        <p>Choose your LLM provider in the sidebar and start chatting!</p>
        <p style="font-size: 0.9rem; margin-top: 1rem; color: #64748b;">
            <strong>Providers:</strong><br>
            🦙 <strong>Ollama</strong> - Local models (free)<br>
            ⚡ <strong>Groq</strong> - Fast cloud inference<br>
            🤖 <strong>OpenAI</strong> - GPT models
        </p>
        <p style="font-size: 0.9rem; margin-top: 1rem; color: #64748b;">
            Add custom tools in <code style="background: rgba(124, 58, 237, 0.2); padding: 2px 8px; border-radius: 4px;">chatbot/tools.py</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
