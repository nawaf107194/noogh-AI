#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Service - The Mouth & Ears of Noogh
==========================================

Provides local Text-to-Speech and Speech-to-Text capabilities.
100% on-device processing - no external APIs.

Features:
- Neural TTS with RealtimeTTS (Coqui engine)
- Local STT with faster-whisper (Whisper small model)
- Fallback to pyttsx3 for lightweight TTS
- Thread-safe audio playback
- GPU acceleration when available
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Union
from threading import Lock
import io
import tempfile

logger = logging.getLogger(__name__)


class VoiceService:
    """
    Singleton service for voice input/output.
    
    Features:
    - Text-to-Speech (TTS) with neural voice
    - Speech-to-Text (STT) with Whisper
    - Thread-safe operation
    - GPU acceleration support
    
    Usage:
        voice = VoiceService()
        text = voice.transcribe(audio_bytes)
        voice.speak("Hello, I am Noogh")
    """
    
    _instance: Optional['VoiceService'] = None
    _lock: Lock = Lock()
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Voice Service."""
        if not self._initialized:
            self._init_services()
            self._initialized = True
    
    def _init_services(self):
        """Initialize TTS and STT engines."""
        logger.info("🎤 Initializing Voice Service...")
        
        self.tts_engine = None
        self.stt_model = None
        self.tts_type = None  # "neural" or "fallback"
        
        # Try to initialize neural TTS first
        self._init_tts()
        
        # Initialize STT
        self._init_stt()
        
        logger.info(f"✅ Voice Service initialized (TTS: {self.tts_type}, STT: {'faster-whisper' if self.stt_model else 'unavailable'})")
    
    def _init_tts(self):
        """Initialize Text-to-Speech engine."""
        # Try RealtimeTTS with Coqui first
        try:
            from RealtimeTTS import TextToAudioStream, CoquiEngine
            
            logger.info("Attempting to load CoquiEngine (Neural TTS)...")
            
            engine = CoquiEngine()
            self.tts_engine = TextToAudioStream(engine)
            self.tts_type = "neural"
            
            logger.info("✅ Neural TTS (Coqui) loaded successfully")
            return
        except ImportError:
            logger.warning("⚠️ RealtimeTTS not installed")
        except Exception as e:
            logger.warning(f"⚠️ CoquiEngine failed to load: {e}")
        
        # Fallback to pyttsx3 (lightweight, system TTS)
        try:
            import pyttsx3
            
            logger.info("Loading fallback TTS (pyttsx3)...")
            
            self.tts_engine = pyttsx3.init()
            
            # Configure voice (try to use a good quality voice)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voices for clarity
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Speed (default ~200)
            self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            self.tts_type = "fallback"
            
            logger.info("✅ Fallback TTS (pyttsx3) loaded successfully")
            return
        except Exception as e:
            logger.error(f"❌ Failed to initialize fallback TTS: {e}")
        
        logger.error("❌ No TTS engine available")
    
    def _init_stt(self):
        """Initialize Speech-to-Text engine."""
        try:
            from faster_whisper import WhisperModel
            
            logger.info("Loading Whisper model (small)...")
            
            # Use small model for speed, tiny for even faster
            # Options: tiny, base, small, medium, large-v2
            model_size = "small"
            
            # GPU if available
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                device = "cpu"
            
            # Load model
            self.stt_model = WhisperModel(
                model_size,
                device=device,
                compute_type="float16" if device == "cuda" else "int8"
            )
            
            logger.info(f"✅ Whisper ({model_size}) loaded on {device}")
        except ImportError:
            logger.error("❌ faster-whisper not installed")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
    
    def transcribe(self, audio_data: Union[bytes, str, Path]) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_data: Audio bytes, file path, or Path object
        
        Returns:
            Transcribed text
        """
        if not self.stt_model:
            logger.error("STT model not available")
            return "[Voice input unavailable]"
        
        try:
            # Handle different input types
            if isinstance(audio_data, bytes):
                # Save bytes to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    audio_path = f.name
            else:
                audio_path = str(audio_data)
            
            # Transcribe
            segments, info = self.stt_model.transcribe(
                audio_path,
                language="en",  # Auto-detect or specify language
                beam_size=5,
                vad_filter=True  # Voice activity detection
            )
            
            # Combine segments
            text = " ".join([segment.text for segment in segments])
            
            # Cleanup temp file if we created one
            if isinstance(audio_data, bytes):
                try:
                    Path(audio_path).unlink()
                except:
                    pass
            
            logger.info(f"🎤 Transcribed: {text[:50]}...")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"[Transcription error: {str(e)}]"
    
    def speak(self, text: str, blocking: bool = False) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        
        Returns:
            True if successful
        """
        if not self.tts_engine:
            logger.error("TTS engine not available")
            return False
        
        try:
            if self.tts_type == "neural":
                # RealtimeTTS (streaming)
                self._speak_neural(text, blocking)
            else:
                # pyttsx3 (fallback)
                self._speak_fallback(text, blocking)
            
            return True
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    def _speak_neural(self, text: str, blocking: bool):
        """Speak using neural TTS (RealtimeTTS)."""
        if blocking:
            # Play synchronously
            self.tts_engine.feed(text).play()
        else:
            # Play in background thread
            def _play():
                self.tts_engine.feed(text).play()
            
            thread = threading.Thread(target=_play, daemon=True)
            thread.start()
    
    def _speak_fallback(self, text: str, blocking: bool):
        """Speak using fallback TTS (pyttsx3)."""
        if blocking:
            # Play synchronously
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        else:
            # Play in background thread
            def _play():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=_play, daemon=True)
            thread.start()
    
    def get_status(self) -> dict:
        """Get voice service status."""
        return {
            "tts_available": self.tts_engine is not None,
            "tts_type": self.tts_type,
            "stt_available": self.stt_model is not None,
            "stt_model": "faster-whisper" if self.stt_model else None
        }


# Convenience functions
def transcribe_audio(audio_data: Union[bytes, str, Path]) -> str:
    """Transcribe audio to text."""
    service = VoiceService()
    return service.transcribe(audio_data)


def speak_text(text: str, blocking: bool = False) -> bool:
    """Convert text to speech."""
    service = VoiceService()
    return service.speak(text, blocking)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "VoiceService",
    "transcribe_audio",
    "speak_text"
]


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("🎤 Voice Service Test")
    print("="*70 + "\n")
    
    # Initialize
    voice = VoiceService()
    status = voice.get_status()
    
    print(f"TTS Available: {status['tts_available']} ({status['tts_type']})")
    print(f"STT Available: {status['stt_available']} ({status['stt_model']})")
    print()
    
    # Test TTS
    if status['tts_available']:
        print("Testing TTS: 'System Online'")
        voice.speak("System Online", blocking=True)
        print("✅ TTS test complete")
    else:
        print("⚠️ TTS not available")
    
    print("\n" + "="*70)
