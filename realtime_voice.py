"""
Realtime Voice Chat - Natural Conversation with Tools.

Continuous voice conversation like ChatGPT's voice mode.
Supports tool calling for hotel services, reservations, etc.
"""

import os
import queue
import threading
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

load_dotenv()

# Import audio library
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from chatbot.realtime_client import RealtimeClient, RealtimeConfig, RealtimeTool
from chatbot.tools import tools_registry

# Page configuration
st.set_page_config(
    page_title="Voice Chat",
    page_icon="🎙️",
    layout="centered",
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%);
    }

    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 2rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #6b7280;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    .orb-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .voice-orb {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        box-shadow: 0 0 40px rgba(75, 85, 99, 0.3);
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 3rem;
        transition: all 0.3s ease;
    }

    .voice-orb.listening {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        box-shadow: 0 0 60px rgba(124, 58, 237, 0.5);
        animation: pulse 2s ease-in-out infinite;
    }

    .voice-orb.speaking {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        box-shadow: 0 0 60px rgba(16, 185, 129, 0.5);
        animation: speak 0.4s ease-in-out infinite;
    }

    .voice-orb.tool {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        box-shadow: 0 0 60px rgba(245, 158, 11, 0.5);
        animation: tool-pulse 0.8s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }

    @keyframes speak {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    @keyframes tool-pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }

    .status-text {
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-size: 1rem;
        margin: 1rem 0;
        min-height: 2rem;
    }

    .status-idle { color: #6b7280; }
    .status-listening { color: #a78bfa; }
    .status-speaking { color: #34d399; }
    .status-tool { color: #fbbf24; }
    .status-error { color: #f87171; }

    .live-text {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #9ca3af;
        font-style: italic;
        padding: 0.5rem 1rem;
        background: rgba(55, 65, 81, 0.3);
        border-radius: 8px;
        margin: 0.5rem auto;
        max-width: 80%;
    }

    .message-container {
        max-height: 300px;
        overflow-y: auto;
        padding: 0.5rem;
    }

    .message {
        font-family: 'Inter', sans-serif;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 16px;
        max-width: 80%;
        line-height: 1.4;
    }

    .user-msg {
        background: rgba(124, 58, 237, 0.2);
        border: 1px solid rgba(124, 58, 237, 0.3);
        color: #e0e0e0;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }

    .assistant-msg {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #e0e0e0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }

    .tool-badge {
        display: inline-block;
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
    }

    .empty-state {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================================
# CONVERT LANGCHAIN TOOLS TO REALTIME FORMAT
# ================================================================================

def convert_langchain_tools_to_realtime() -> List[RealtimeTool]:
    """Convert registered LangChain tools to Realtime API format."""
    realtime_tools = []
    
    for lc_tool in tools_registry.get_tools():
        # Extract the JSON schema from LangChain tool (use model_json_schema for Pydantic v2)
        if lc_tool.args_schema:
            try:
                schema = lc_tool.args_schema.model_json_schema()
            except AttributeError:
                # Fallback for older Pydantic
                schema = lc_tool.args_schema.schema()
        else:
            schema = {"type": "object", "properties": {}}
        
        # Build parameters in OpenAI format
        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        
        realtime_tool = RealtimeTool(
            name=lc_tool.name,
            description=lc_tool.description,
            parameters=parameters,
            handler=lc_tool.func,
        )
        realtime_tools.append(realtime_tool)
        
    return realtime_tools


# ================================================================================
# THREAD-SAFE STATE & GLOBAL REFERENCES
# ================================================================================

@dataclass
class ConversationState:
    """Thread-safe state for the conversation."""
    client: Optional[RealtimeClient] = None
    audio_manager: Optional[Any] = None  # Reference to AudioManager
    is_active: bool = False
    is_ai_speaking: bool = False
    is_using_tool: bool = False
    messages: List[dict] = field(default_factory=list)
    current_text: str = ""
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_message(self, role: str, text: str):
        with self._lock:
            if text.strip():
                self.messages.append({"role": role, "text": text.strip()})
    
    def get_messages(self) -> List[dict]:
        with self._lock:
            return self.messages.copy()
    
    def clear_messages(self):
        with self._lock:
            self.messages = []


# Initialize state once
if "conv_state" not in st.session_state:
    st.session_state.conv_state = ConversationState()

# Get reference (this is safe - the object persists)
state: ConversationState = st.session_state.conv_state


# ================================================================================
# BUFFERED AUDIO OUTPUT (Fixes stuttering)
# ================================================================================

class BufferedAudioOutput:
    """
    Buffered audio output stream to prevent stuttering.
    Accumulates audio chunks and plays them smoothly.
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.buffer_lock = threading.Lock()
        self.stream = None
        self.is_playing = False
        self._stop = False
        self.play_thread = None
    
    def add_audio(self, audio_bytes: bytes):
        """Add audio chunk to buffer."""
        with self.buffer_lock:
            self.buffer.extend(audio_bytes)
    
    def start(self):
        """Start the audio output thread."""
        self._stop = False
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()
    
    def _play_loop(self):
        """Continuous playback loop."""
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=2048,
            )
            self.stream.start()
            
            while not self._stop:
                with self.buffer_lock:
                    if len(self.buffer) >= 4096:
                        chunk_size = min(len(self.buffer), 8192)
                        chunk = bytes(self.buffer[:chunk_size])
                        del self.buffer[:chunk_size]
                    else:
                        chunk = None
                
                if chunk:
                    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
                    self.stream.write(audio)
                    self.is_playing = True
                else:
                    self.is_playing = False
                    time.sleep(0.01)
            
        except Exception as e:
            print(f"Audio output error: {e}")
        finally:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
    
    def stop(self):
        """Stop playback and clear buffer."""
        self._stop = True
        if self.play_thread:
            self.play_thread.join(timeout=1)
        with self.buffer_lock:
            self.buffer.clear()
    
    def clear(self):
        """Clear the buffer."""
        with self.buffer_lock:
            self.buffer.clear()


# ================================================================================
# AUDIO MANAGER
# ================================================================================

class AudioManager:
    """Manages microphone input and speaker output."""
    
    def __init__(self, conv_state: ConversationState):
        self.state = conv_state
        self.input_stream = None
        self.output = BufferedAudioOutput()
        self._stop = False
    
    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone callback - sends audio to API."""
        if self.state.client and self.state.is_active and not self.state.is_ai_speaking:
            try:
                audio = (indata[:, 0] * 32767).astype(np.int16)
                self.state.client.send_audio(audio.tobytes())
            except Exception:
                pass
    
    def add_output_audio(self, audio_bytes: bytes):
        """Add audio to the output buffer."""
        self.output.add_audio(audio_bytes)
    
    def start(self) -> bool:
        """Start audio streams."""
        try:
            self.output.start()
            
            self.input_stream = sd.InputStream(
                samplerate=24000,
                channels=1,
                dtype=np.float32,
                callback=self._mic_callback,
                blocksize=2400,
            )
            self.input_stream.start()
            
            return True
        except Exception as e:
            self.state.error = f"Audio error: {e}"
            return False
    
    def stop(self):
        """Stop audio streams."""
        self.output.stop()
        
        if self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except:
                pass
            self.input_stream = None


# ================================================================================
# CALLBACKS (Use state object directly, NOT session_state)
# ================================================================================

def on_audio_chunk(audio_bytes: bytes):
    """Receive audio from AI - add to buffer."""
    # Use the state.audio_manager reference directly (thread-safe)
    if state.audio_manager:
        state.audio_manager.add_output_audio(audio_bytes)


def on_transcript_delta(role: str, text: str):
    """Live transcript updates."""
    if role == "assistant":
        with state._lock:
            state.current_text += text


def on_transcript_complete(role: str, text: str):
    """Complete transcript received."""
    state.add_message(role, text)
    if role == "assistant":
        with state._lock:
            state.current_text = ""


def on_ai_speaking():
    """AI started speaking."""
    state.is_ai_speaking = True
    state.is_using_tool = False
    with state._lock:
        state.current_text = ""


def on_ai_stopped():
    """AI stopped speaking."""
    state.is_ai_speaking = False


def on_error(error: str):
    """Handle errors."""
    state.error = error
    state.is_active = False


# ================================================================================
# UI
# ================================================================================

st.markdown('<h1 class="main-header">🎙️ Voice Chat</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Natural conversation with hotel assistant</p>', unsafe_allow_html=True)


# ================================================================================
# SIDEBAR
# ================================================================================

with st.sidebar:
    st.markdown("### Settings")
    
    voice = st.selectbox("Voice", ["nova", "alloy", "echo", "fable", "onyx", "shimmer"], index=0)
    
    instructions = st.text_area(
        "AI Personality",
        value="""You are a friendly hotel concierge assistant. Help guests with:
- Room service requests (towels, water, amenities)
- Reservation lookups
- Local recommendations
- General hotel information

Be warm, professional, and concise. Use the available tools when guests need services.""",
        height=120,
    )
    
    st.divider()
    
    sensitivity = st.slider("Voice Sensitivity", 0.3, 0.8, 0.5)
    response_delay = st.slider("Response Delay (ms)", 400, 1000, 600)
    
    st.divider()
    
    # Show available tools
    st.markdown("### 🛠️ Available Tools")
    tool_names = tools_registry.list_tools()
    if tool_names:
        for name in tool_names:
            st.markdown(f'<span class="tool-badge">{name}</span>', unsafe_allow_html=True)
    else:
        st.caption("No tools registered")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        state.clear_messages()
        st.rerun()


# ================================================================================
# API KEY CHECK
# ================================================================================

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    st.error("⚠️ Set OPENAI_API_KEY in your .env file")
    st.stop()

if not AUDIO_AVAILABLE:
    st.error("⚠️ Install sounddevice: `pip install sounddevice`")
    st.stop()


# ================================================================================
# VOICE ORB & STATUS
# ================================================================================

orb_class = "voice-orb"
status_class = "status-idle"
status_msg = "Click below to start"
icon = "🎤"

if state.is_active:
    if state.is_using_tool:
        orb_class = "voice-orb tool"
        status_class = "status-tool"
        status_msg = "Using tools..."
        icon = "🔧"
    elif state.is_ai_speaking:
        orb_class = "voice-orb speaking"
        status_class = "status-speaking"
        status_msg = "Speaking..."
        icon = "🔊"
    else:
        orb_class = "voice-orb listening"
        status_class = "status-listening"
        status_msg = "Listening..."
        icon = "🎤"

if state.error:
    status_class = "status-error"
    status_msg = state.error

st.markdown(f'''
<div class="orb-container">
    <div class="{orb_class}">{icon}</div>
</div>
<p class="status-text {status_class}">{status_msg}</p>
''', unsafe_allow_html=True)

# Live transcript (thread-safe read)
with state._lock:
    current_text = state.current_text

if current_text:
    st.markdown(f'<div class="live-text">"{current_text}"</div>', unsafe_allow_html=True)


# ================================================================================
# CONTROL BUTTON
# ================================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if not state.is_active:
        if st.button("🎙️ Start Talking", use_container_width=True, type="primary"):
            state.error = ""
            
            # Convert tools to Realtime format
            realtime_tools = convert_langchain_tools_to_realtime()
            print(f"\n📦 Loaded {len(realtime_tools)} tools for Realtime API:")
            for t in realtime_tools:
                print(f"   - {t.name}: {t.description[:50]}...")
            
            config = RealtimeConfig(
                voice=voice,
                instructions=instructions,
                turn_detection={
                    "type": "server_vad",
                    "threshold": sensitivity,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": response_delay,
                },
                tools=realtime_tools,
            )
            
            client = RealtimeClient(
                api_key=api_key,
                config=config,
                on_audio=on_audio_chunk,
                on_transcript=on_transcript_delta,
                on_transcript_done=on_transcript_complete,
                on_speech_started=on_ai_speaking,
                on_speech_stopped=on_ai_stopped,
                on_error=on_error,
            )
            
            with st.spinner("Connecting..."):
                client.start()
                time.sleep(1.5)
                
                if client.is_connected:
                    state.client = client
                    
                    # Create and store audio manager in state (NOT session_state)
                    audio = AudioManager(state)
                    if audio.start():
                        state.audio_manager = audio  # Store in state object
                        state.is_active = True
                        st.rerun()
                    else:
                        client.stop()
                else:
                    state.error = "Connection failed. Check API key."
                    st.rerun()
    else:
        if st.button("⏹️ Stop", use_container_width=True):
            # Stop audio manager
            if state.audio_manager:
                state.audio_manager.stop()
                state.audio_manager = None
            
            # Stop client
            if state.client:
                state.client.stop()
                state.client = None
            
            state.is_active = False
            state.is_ai_speaking = False
            with state._lock:
                state.current_text = ""
            st.rerun()


# ================================================================================
# CONVERSATION
# ================================================================================

st.markdown("---")
st.markdown("### Conversation")

messages = state.get_messages()

if messages:
    for msg in messages:
        cls = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f'<div class="message {cls}">{msg["text"]}</div>', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="empty-state">
        <p>Your conversation will appear here.</p>
        <p style="font-size: 0.85rem;">Try saying: "I need some towels in room 1000"</p>
    </div>
    ''', unsafe_allow_html=True)


# ================================================================================
# AUTO-REFRESH
# ================================================================================

if state.is_active:
    time.sleep(0.3)
    st.rerun()
