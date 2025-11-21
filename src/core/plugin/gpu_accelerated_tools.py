"""
GPU Accelerated Tools - Mock Implementation
This module provides mock implementations for GPU tools when actual GPU libraries are not available.
"""

import logging
import platform

logger = logging.getLogger(__name__)

# Constants
USE_GPU = False
DEVICE = "cpu"
DEVICE_STR = "CPU (Mock)"

def get_gpu_info():
    """Get GPU information"""
    return {
        "device": DEVICE,
        "name": "CPU Fallback",
        "memory_total": 0,
        "memory_allocated": 0,
        "memory_reserved": 0
    }

def benchmark_gpu():
    """Run a mock benchmark"""
    return {
        "score": 0,
        "device": DEVICE,
        "message": "Benchmark skipped (No GPU)"
    }

class StableDiffusionGenerator:
    async def generate_image(self, prompt, **kwargs):
        logger.warning(f"Mock generating image for: {prompt}")
        return ["mock_image_url"]

class WhisperTranscriber:
    async def transcribe(self, audio_path, **kwargs):
        logger.warning(f"Mock transcribing: {audio_path}")
        return {"text": "Mock transcription result"}

class CLIPImageAnalyzer:
    async def analyze_image(self, image_path, **kwargs):
        logger.warning(f"Mock analyzing image: {image_path}")
        return {"labels": ["mock_label"], "scores": [0.99]}

class EmbeddingsEngine:
    async def compute_embeddings(self, texts):
        return [[0.1] * 768 for _ in texts]

    async def find_similar(self, query, corpus, top_k=5):
        return [{"text": c, "score": 0.5} for c in corpus[:top_k]]

class CodeGenerator:
    async def generate_code(self, prompt, **kwargs):
        return f"# Mock code for: {prompt}\ndef mock_function():\n    pass"

class NeuralTranslator:
    async def translate(self, text, source_lang, target_lang):
        return f"[Translated to {target_lang}]: {text}"

class ObjectDetector:
    async def detect_objects(self, image_path, confidence=0.25):
        return [{"label": "mock_object", "confidence": 0.9, "box": [0, 0, 100, 100]}]
