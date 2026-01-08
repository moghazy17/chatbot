"""
Streamlit Chatbot Application.

A modern chatbot interface powered by LangGraph with support for
multiple LLM providers and interaction modes:
- Text Chat: Standard text messaging
- Voice Chat: Record → Transcribe → Respond → Speak
- Realtime Voice: Continuous hands-free conversation
"""

import os
import streamlit as st
from dotenv import load_dotenv

from chatbot import TextChatHandler, VoiceChatHandler
from chatbot.tools import tools_registry
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

    .mode-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 4px;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
    }

    .mode-text { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
    .mode-voice { background: linear-gradient(135deg, #ec4899, #db2777); color: white; }

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

    .tab-content {
        padding: 1rem 0;
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

# ================================================================================
# SIDEBAR CONFIGURATION
# ================================================================================

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
        default_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        base_url = st.text_input(
            "Ollama URL",
            value=default_url,
            placeholder="http://localhost:11434",
            help="Enter your Ollama server URL"
        )

        default_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_models = LLMFactory.get_default_models(LLMProvider.OLLAMA)
        model = st.selectbox(
            "Model",
            ollama_models,
            index=ollama_models.index(default_model) if default_model in ollama_models else 0,
            help="Select the Ollama model to use"
        )

        custom_model = st.text_input(
            "Or enter custom model",
            placeholder="e.g., llama3.2:70b",
            help="Enter a specific model name if not in the list"
        )
        selected_model = custom_model if custom_model else model

    elif provider == "groq":
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

        default_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        groq_models = LLMFactory.get_default_models(LLMProvider.GROQ)
        selected_model = st.selectbox(
            "Model",
            groq_models,
            index=groq_models.index(default_model) if default_model in groq_models else 0,
            help="Select the Groq model to use"
        )

    elif provider == "openai":
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

        default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_models = LLMFactory.get_default_models(LLMProvider.OPENAI)
        selected_model = st.selectbox(
            "Model",
            openai_models,
            index=openai_models.index(default_model) if default_model in openai_models else 0,
            help="Select the OpenAI model to use"
        )

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

    # Audio settings section
    st.markdown("### 🎤 Voice Settings")

    tts_voice = st.selectbox(
        "TTS Voice",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        index=0,
        help="Select the voice for text-to-speech output"
    )

    tts_enabled = st.toggle("Enable TTS Response", value=True, help="Convert AI responses to speech")

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

    # Realtime voice mode link
    st.markdown("### 🎙️ Realtime Voice")
    st.markdown("""
    <p style="font-size: 0.85rem; color: #94a3b8;">
        For real-time voice conversations:
    </p>
    <code style="background: rgba(124, 58, 237, 0.2); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;">
        streamlit run realtime_voice.py
    </code>
    """, unsafe_allow_html=True)

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.text_handler = None
        st.session_state.voice_handler = None
        st.rerun()


# ================================================================================
# SESSION STATE INITIALIZATION
# ================================================================================

def get_text_handler() -> TextChatHandler:
    """Get or create the text chat handler."""
    handler_key = f"text_handler_{provider}_{selected_model}_{temperature}"
    
    if "text_handler" not in st.session_state or st.session_state.get("text_handler_key") != handler_key:
        handler = TextChatHandler(
            provider=provider,
            model=selected_model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
        if handler.initialize():
            st.session_state.text_handler = handler
            st.session_state.text_handler_key = handler_key
        else:
            st.error("Failed to initialize text chat handler")
            return None
    
    return st.session_state.text_handler


def get_voice_handler() -> VoiceChatHandler:
    """Get or create the voice chat handler."""
    handler_key = f"voice_handler_{provider}_{selected_model}_{temperature}_{tts_voice}"
    
    if "voice_handler" not in st.session_state or st.session_state.get("voice_handler_key") != handler_key:
        handler = VoiceChatHandler(
            provider=provider,
            model=selected_model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            tts_voice=tts_voice,
        )
        if handler.initialize():
            st.session_state.voice_handler = handler
            st.session_state.voice_handler_key = handler_key
        else:
            st.error("Failed to initialize voice chat handler")
            return None
    
    return st.session_state.voice_handler


# ================================================================================
# CHAT MODE TABS
# ================================================================================

# Create tabs for different modes
tab_text, tab_voice = st.tabs(["💬 Text Chat", "🎤 Voice Chat"])

# ================================================================================
# TEXT CHAT TAB
# ================================================================================

with tab_text:
    st.markdown('<span class="mode-badge mode-text">💬 Text Mode</span>', unsafe_allow_html=True)
    
    # Get the handler
    text_handler = get_text_handler()
    
    if text_handler:
        # Display chat history
        for message in text_handler.messages:
            avatar = "👤" if message.role == "user" else "🤖"
            with st.chat_message(message.role, avatar=avatar):
                st.write(message.content)
        
        # Chat input
        if prompt := st.chat_input("Type your message...", key="text_input"):
            if provider in ["groq", "openai"] and not api_key:
                st.error(f"Please enter your {provider.upper()} API key in the sidebar.")
            else:
                # Display user message
                with st.chat_message("user", avatar="👤"):
                    st.write(prompt)
                
                # Get response
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        try:
                            response = text_handler.send_message(prompt)
                            st.write(response.content)
                            
                            # Generate TTS if enabled
                            if tts_enabled and response.content:
                                voice_handler = get_voice_handler()
                                if voice_handler:
                                    with st.spinner("Generating speech..."):
                                        try:
                                            audio_bytes, audio_format = voice_handler.speak(response.content)
                                            if audio_bytes:
                                                st.audio(audio_bytes, format=f"audio/{audio_format}")
                                        except Exception as e:
                                            st.warning(f"TTS unavailable: {str(e)}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Show welcome message if no messages
        if not text_handler.messages:
            st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem;
                color: #94a3b8;
                font-family: 'Outfit', sans-serif;
            ">
                <p style="font-size: 1.2rem; margin-bottom: 1rem;">👋 Welcome to Text Chat!</p>
                <p>Type a message below to start chatting.</p>
            </div>
            """, unsafe_allow_html=True)


# ================================================================================
# VOICE CHAT TAB
# ================================================================================

with tab_voice:
    st.markdown('<span class="mode-badge mode-voice">🎤 Voice Mode</span>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
        Record your voice → AI transcribes → Responds with text and speech
    </p>
    """, unsafe_allow_html=True)
    
    # Get the handler
    voice_handler = get_voice_handler()
    
    if voice_handler:
        # Display chat history
        for message in voice_handler.messages:
            avatar = "👤" if message.role == "user" else "🤖"
            with st.chat_message(message.role, avatar=avatar):
                if message.role == "user":
                    st.write(f"🎤 {message.content}")
                else:
                    st.write(message.content)
        
        # Audio input
        st.markdown("### 🎙️ Record Your Message")
        audio_input = st.audio_input("Click to record", key="voice_recorder")
        
        if audio_input is not None:
            audio_bytes = audio_input.read()
            
            if audio_bytes:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.audio(audio_bytes, format="audio/wav")
                
                with col2:
                    if st.button("📤 Send", use_container_width=True, key="send_voice"):
                        if provider in ["groq", "openai"] and not api_key:
                            st.error(f"Please enter your {provider.upper()} API key in the sidebar.")
                        else:
                            with st.spinner("Processing..."):
                                try:
                                    # Process audio through voice handler
                                    response, audio_response = voice_handler.process_audio(
                                        audio_bytes,
                                        filename="audio.wav",
                                        generate_audio=tts_enabled,
                                    )
                                    
                                    # Display the transcribed user message
                                    with st.chat_message("user", avatar="👤"):
                                        # Get the last user message
                                        user_msgs = [m for m in voice_handler.messages if m.role == "user"]
                                        if user_msgs:
                                            st.write(f"🎤 {user_msgs[-1].content}")
                                    
                                    # Display response
                                    with st.chat_message("assistant", avatar="🤖"):
                                        st.write(response.content)
                                        
                                        # Play audio response
                                        if audio_response:
                                            audio_bytes_out, audio_format = audio_response
                                            st.audio(audio_bytes_out, format=f"audio/{audio_format}")
                                    
                                    st.rerun()
                                    
                                except ValueError as e:
                                    st.warning(f"Could not process audio: {str(e)}")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
        
        # Show welcome message if no messages
        if not voice_handler.messages:
            st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem;
                color: #94a3b8;
                font-family: 'Outfit', sans-serif;
            ">
                <p style="font-size: 1.2rem; margin-bottom: 1rem;">👋 Welcome to Voice Chat!</p>
                <p>Click the microphone button above to record your message.</p>
                <p style="font-size: 0.85rem; margin-top: 1rem;">
                    Your speech will be transcribed and the AI will respond with text and audio.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ================================================================================
# FOOTER - REALTIME MODE INFO
# ================================================================================

st.markdown("---")
st.markdown("""
<div style="
    text-align: center;
    padding: 1rem;
    color: #64748b;
    font-family: 'Outfit', sans-serif;
">
    <p style="font-size: 0.95rem;">
        🎙️ <strong>Want Real-Time Voice?</strong>
    </p>
    <p style="font-size: 0.85rem;">
        For continuous, hands-free voice conversation with automatic speech detection,<br>
        run <code style="background: rgba(124, 58, 237, 0.2); padding: 2px 8px; border-radius: 4px;">streamlit run realtime_voice.py</code>
    </p>
</div>
""", unsafe_allow_html=True)
