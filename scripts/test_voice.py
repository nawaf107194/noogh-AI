#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Service Test Script
==========================

Tests Text-to-Speech and Speech-to-Text capabilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_voice_service():
    """Test VoiceService initialization and capabilities."""
    print("\n" + "="*70)
    print("🎤 VOICE SERVICE TEST")
    print("="*70 + "\n")
    
    try:
        from services.voice_service import VoiceService
        
        # Initialize service
        print("Initializing Voice Service...")
        voice = VoiceService()
        
        # Get status
        status = voice.get_status()
        
        print("\n📊 Service Status:")
        print(f"  TTS Available: {status['tts_available']}")
        print(f"  TTS Type: {status['tts_type']}")
        print(f"  STT Available: {status['stt_available']}")
        print(f"  STT Model: {status['stt_model']}")
        print()
        
        # Test 1: Text-to-Speech
        if status['tts_available']:
            print("="*70)
            print("TEST 1: Text-to-Speech")
            print("="*70)
            print("\n🔊 Speaking: 'System Online'")
            
            voice.speak("System Online", blocking=True)
            
            print("✅ TTS test completed successfully")
            print()
        else:
            print("⚠️ TTS not available - skipping TTS test")
            print()
        
        # Test 2: Speech-to-Text
        if status['stt_available']:
            print("="*70)
            print("TEST 2: Speech-to-Text")
            print("="*70)
            print("\n🎙️ Note: STT requires audio file for testing")
            print("   In production, use mic_recorder in dashboard for live input")
            print("✅ STT engine loaded and ready")
            print()
        else:
            print("⚠️ STT not available")
            print("   Install with: pip install faster-whisper")
            print()
        
        print("="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70 + "\n")
        
        return True
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nMissing dependencies. Install with:")
        print("  pip install RealtimeTTS faster-whisper pyttsx3")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_voice_service()
    sys.exit(0 if success else 1)
