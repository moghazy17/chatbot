"""
Realtime Voice Chat Application.

Real-time voice conversation interface using OpenAI Realtime API.
"""

import os
import queue
import threading
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

from chatbot.realtime_client import RealtimeClient, RealtimeConfig

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Voice Chat",
    page_icon="🎙️",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');

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

    .transcript-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Outfit', sans-serif;
    }

    .user-msg {
        color: #60a5fa;
        border-left: 3px solid #60a5fa;
        padding-left: 1rem;
    }

    .assistant-msg {
        color: #a78bfa;
        border-left: 3px solid #a78bfa;
        padding-left: 1rem;
    }

    .status-connected {
        color: #4ade80;
        font-weight: 500;
    }

    .status-disconnected {
        color: #f87171;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎙️ Realtime Voice Chat</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Talk naturally with AI using OpenAI Realtime API</p>', unsafe_allow_html=True)


# Initialize session state
if "realtime_client" not in st.session_state:
    st.session_state.realtime_client = None
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []
if "is_connected" not in st.session_state:
    st.session_state.is_connected = False


class AudioProcessor(AudioProcessorBase):
    """Process audio from WebRTC and send to Realtime API."""

    def __init__(self):
        self.client = None
        self.output_queue = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Receive audio frame from microphone."""
        # Convert to numpy array
        audio = frame.to_ndarray()

        # Resample to 24kHz if needed (OpenAI Realtime requires 24kHz)
        if frame.sample_rate != 24000:
            # Simple resampling - in production use proper resampling
            ratio = 24000 / frame.sample_rate
            new_length = int(len(audio[0]) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio[0]), new_length),
                np.arange(len(audio[0])),
                audio[0]
            ).astype(np.int16)
        else:
            audio = audio[0].astype(np.int16)

        # Send to Realtime API
        if self.client and self.client.is_connected:
            self.client.send_audio(audio.tobytes())

        # Check for output audio
        output_audio = None
        try:
            output_audio = self.output_queue.get_nowait()
        except queue.Empty:
            pass

        if output_audio is not None:
            # Create output frame with received audio
            output_array = np.frombuffer(output_audio, dtype=np.int16)
            output_frame = av.AudioFrame.from_ndarray(
                output_array.reshape(1, -1),
                format='s16',
                layout='mono'
            )
            output_frame.sample_rate = 24000
            return output_frame

        # Return silence if no output
        silence = np.zeros((1, frame.samples), dtype=np.int16)
        output_frame = av.AudioFrame.from_ndarray(silence, format='s16', layout='mono')
        output_frame.sample_rate = frame.sample_rate
        return output_frame


# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Voice Settings")

    voice = st.selectbox(
        "Voice",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        index=0,
        help="Select the AI voice"
    )

    st.divider()

    instructions = st.text_area(
        "System Instructions",
        value="You are a helpful hotel assistant. Help guests with reservations, amenities, and local recommendations. Be friendly and conversational.",
        height=100,
        help="Instructions for the AI assistant"
    )

    st.divider()

    # Connection status
    if st.session_state.is_connected:
        st.markdown('<p class="status-connected">🟢 Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-disconnected">🔴 Disconnected</p>', unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.transcripts = []
        st.rerun()


# Callbacks for Realtime client
def on_audio_received(audio_bytes: bytes):
    """Handle audio output from the API."""
    if "audio_processor" in st.session_state and st.session_state.audio_processor:
        st.session_state.audio_processor.output_queue.put(audio_bytes)


def on_transcript_received(role: str, text: str):
    """Handle transcript from the API."""
    st.session_state.transcripts.append({"role": role, "text": text})


def on_error(error: str):
    """Handle errors from the API."""
    st.error(f"Realtime API Error: {error}")


# Main content area
st.markdown("### 💬 Conversation")

# Display transcripts
transcript_container = st.container()
with transcript_container:
    for msg in st.session_state.transcripts:
        role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="transcript-box {role_class}">{role_icon} {msg["text"]}</div>',
            unsafe_allow_html=True
        )

st.markdown("---")

# Check for API key
api_key = os.getenv("OPEN_API_KEY", "").strip()
if not api_key:
    st.error("⚠️ OPEN_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

# WebRTC streamer
st.markdown("### 🎤 Voice Input")
st.info("Click START to begin the conversation. Speak naturally - the AI will respond in real-time.")

webrtc_ctx = webrtc_streamer(
    key="realtime-voice",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={
        "audio": {
            "sampleRate": 24000,
            "channelCount": 1,
            "echoCancellation": True,
            "noiseSuppression": True,
        },
        "video": False,
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True,
)

# Manage Realtime client connection
if webrtc_ctx.state.playing:
    if not st.session_state.is_connected:
        # Create and start Realtime client
        try:
            config = RealtimeConfig(
                voice=voice,
                instructions=instructions,
            )
            client = RealtimeClient(
                config=config,
                on_audio=on_audio_received,
                on_transcript=on_transcript_received,
                on_error=on_error,
            )
            client.start()
            st.session_state.realtime_client = client
            st.session_state.is_connected = True

            # Connect processor to client
            if webrtc_ctx.audio_processor:
                webrtc_ctx.audio_processor.client = client
                st.session_state.audio_processor = webrtc_ctx.audio_processor

            st.rerun()
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")
else:
    if st.session_state.is_connected:
        # Disconnect
        if st.session_state.realtime_client:
            st.session_state.realtime_client.stop()
            st.session_state.realtime_client = None
        st.session_state.is_connected = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <p>Powered by OpenAI Realtime API | GPT-4o</p>
    <p>Speak naturally and the AI will respond in real-time</p>
</div>
""", unsafe_allow_html=True)
