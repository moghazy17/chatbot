"""
Realtime Voice Chat - Natural Conversation Mode.

Continuous voice conversation like ChatGPT's voice mode.
Just speak naturally and the AI responds automatically.
"""

import os
import queue
import threading
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional, List

load_dotenv()

# Try to import audio library
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from chatbot.realtime_client import RealtimeClient, RealtimeConfig

# Page configuration
st.set_page_config(
    page_title="Voice Chat",
    page_icon="🎙️",
    layout="centered",
)

# Clean, modern CSS
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

    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 60px rgba(124, 58, 237, 0.5); }
        50% { transform: scale(1.03); box-shadow: 0 0 80px rgba(124, 58, 237, 0.7); }
    }

    @keyframes speak {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    .status-text {
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-size: 1rem;
        margin: 1rem 0;
        padding: 0.5rem;
        min-height: 2rem;
    }

    .status-idle { color: #6b7280; }
    .status-listening { color: #a78bfa; }
    .status-speaking { color: #34d399; }
    .status-error { color: #f87171; }

    .live-text {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #9ca3af;
        font-style: italic;
        padding: 0.5rem 1rem;
        min-height: 1.5rem;
        background: rgba(55, 65, 81, 0.3);
        border-radius: 8px;
        margin: 0.5rem auto;
        max-width: 80%;
    }

    .message-container {
        max-height: 350px;
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

    .empty-state {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================================
# THREAD-SAFE STATE (Using class instead of session_state in callbacks)
# ================================================================================

@dataclass
class ConversationState:
    """Thread-safe state shared between audio threads and main app."""
    client: Optional[RealtimeClient] = None
    is_active: bool = False
    is_ai_speaking: bool = False
    audio_queue: queue.Queue = field(default_factory=queue.Queue)
    messages: List[dict] = field(default_factory=list)
    current_text: str = ""
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_message(self, role: str, text: str):
        """Thread-safe message addition."""
        with self._lock:
            if text.strip():
                self.messages.append({"role": role, "text": text.strip()})
    
    def get_messages(self) -> List[dict]:
        """Thread-safe message retrieval."""
        with self._lock:
            return self.messages.copy()
    
    def set_current_text(self, text: str):
        """Thread-safe current text update."""
        with self._lock:
            self.current_text = text
    
    def clear_messages(self):
        """Thread-safe clear."""
        with self._lock:
            self.messages = []


# Initialize state
if "conv_state" not in st.session_state:
    st.session_state.conv_state = ConversationState()

state = st.session_state.conv_state


# ================================================================================
# CALLBACKS (Called from websocket/audio threads)
# ================================================================================

def on_audio_chunk(audio_bytes: bytes):
    """Receive audio from AI."""
    state.audio_queue.put(audio_bytes)


def on_transcript_delta(role: str, text: str):
    """Live transcript updates."""
    if role == "assistant":
        state.set_current_text(state.current_text + text)


def on_transcript_complete(role: str, text: str):
    """Complete transcript received."""
    state.add_message(role, text)
    if role == "assistant":
        state.set_current_text("")


def on_ai_speaking():
    """AI started speaking."""
    state.is_ai_speaking = True
    state.set_current_text("")


def on_ai_stopped():
    """AI stopped speaking."""
    state.is_ai_speaking = False


def on_error(error: str):
    """Handle errors."""
    state.error = error
    state.is_active = False


# ================================================================================
# AUDIO MANAGER
# ================================================================================

class AudioStream:
    """Manages microphone input and speaker output."""
    
    def __init__(self, conv_state: ConversationState):
        self.state = conv_state
        self.input_stream = None
        self.output_thread = None
        self._stop = False
    
    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone callback - sends audio to API."""
        if self.state.client and self.state.is_active:
            try:
                # Convert to int16 PCM at 24kHz
                audio = (indata[:, 0] * 32767).astype(np.int16)
                self.state.client.send_audio(audio.tobytes())
            except Exception:
                pass
    
    def _speaker_worker(self):
        """Speaker output thread - plays AI audio."""
        while not self._stop:
            try:
                chunk = self.state.audio_queue.get(timeout=0.1)
                if chunk:
                    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
                    sd.play(audio, samplerate=24000, blocking=True)
            except queue.Empty:
                pass
            except Exception:
                pass
    
    def start(self) -> bool:
        """Start audio streams."""
        try:
            # Input stream (microphone)
            self.input_stream = sd.InputStream(
                samplerate=24000,
                channels=1,
                dtype=np.float32,
                callback=self._mic_callback,
                blocksize=2400,  # 100ms at 24kHz
            )
            self.input_stream.start()
            
            # Output thread (speaker)
            self._stop = False
            self.output_thread = threading.Thread(target=self._speaker_worker, daemon=True)
            self.output_thread.start()
            
            return True
        except Exception as e:
            self.state.error = f"Audio error: {e}"
            return False
    
    def stop(self):
        """Stop audio streams."""
        self._stop = True
        
        if self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except:
                pass
            self.input_stream = None
        
        if self.output_thread:
            self.output_thread.join(timeout=1)
            self.output_thread = None


# Store audio stream
if "audio_stream" not in st.session_state:
    st.session_state.audio_stream = None


# ================================================================================
# UI HEADER
# ================================================================================

st.markdown('<h1 class="main-header">🎙️ Voice Chat</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Speak naturally. I\'m listening.</p>', unsafe_allow_html=True)


# ================================================================================
# SIDEBAR SETTINGS
# ================================================================================

with st.sidebar:
    st.markdown("### Settings")
    
    voice = st.selectbox(
        "Voice",
        ["nova", "alloy", "echo", "fable", "onyx", "shimmer"],
        index=0,
    )
    
    instructions = st.text_area(
        "AI Personality",
        value="You are a friendly assistant having a natural conversation. Be concise, warm, and helpful. Respond conversationally as if chatting with a friend.",
        height=80,
    )
    
    st.divider()
    
    sensitivity = st.slider("Voice Sensitivity", 0.3, 0.8, 0.5)
    response_delay = st.slider("Response Delay (ms)", 400, 1000, 600)
    
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

# Determine state
orb_class = "voice-orb"
status_class = "status-idle"
status_msg = "Click below to start"

if state.is_active:
    if state.is_ai_speaking:
        orb_class = "voice-orb speaking"
        status_class = "status-speaking"
        status_msg = "Speaking..."
    else:
        orb_class = "voice-orb listening"
        status_class = "status-listening"
        status_msg = "Listening..."

if state.error:
    status_class = "status-error"
    status_msg = state.error

# Display orb
icon = "🎤" if not state.is_ai_speaking else "🔊"
st.markdown(f'''
<div class="orb-container">
    <div class="{orb_class}">{icon}</div>
</div>
<p class="status-text {status_class}">{status_msg}</p>
''', unsafe_allow_html=True)

# Live transcript
if state.current_text:
    st.markdown(f'<div class="live-text">"{state.current_text}"</div>', unsafe_allow_html=True)


# ================================================================================
# CONTROL BUTTON
# ================================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if not state.is_active:
        if st.button("🎙️ Start Talking", use_container_width=True, type="primary"):
            state.error = ""
            
            config = RealtimeConfig(
                voice=voice,
                instructions=instructions,
                turn_detection={
                    "type": "server_vad",
                    "threshold": sensitivity,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": response_delay,
                },
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
                    
                    audio = AudioStream(state)
                    if audio.start():
                        st.session_state.audio_stream = audio
                        state.is_active = True
                        st.rerun()
                    else:
                        client.stop()
                else:
                    state.error = "Connection failed. Check API key."
                    st.rerun()
    else:
        if st.button("⏹️ Stop", use_container_width=True):
            if st.session_state.audio_stream:
                st.session_state.audio_stream.stop()
                st.session_state.audio_stream = None
            
            if state.client:
                state.client.stop()
                state.client = None
            
            state.is_active = False
            state.is_ai_speaking = False
            state.set_current_text("")
            st.rerun()


# ================================================================================
# CONVERSATION HISTORY
# ================================================================================

st.markdown("---")
st.markdown("### Conversation")

messages = state.get_messages()

if messages:
    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    for msg in messages:
        cls = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f'<div class="message {cls}">{msg["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="empty-state">
        <p>Your conversation will appear here.</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem;">
            Just click "Start Talking" and speak naturally!
        </p>
    </div>
    ''', unsafe_allow_html=True)


# ================================================================================
# AUTO-REFRESH WHEN ACTIVE
# ================================================================================

if state.is_active:
    time.sleep(0.3)
    st.rerun()
