#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Brain Service - Sovereign AI with On-Device Inference
============================================================

Service managing local LLM inference using HuggingFace transformers.
NO external APIs - 100% local and private!
"""

import logging
import torch
from typing import Optional
from threading import Lock
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalBrainService:
    """
    Service for local LLM inference.
    
    Uses singleton pattern to load the model once and reuse across all requests.
    Supports both GPU and CPU inference.
    
    Features:
    - Singleton pattern (one model instance)
    - GPU/CPU auto-detection
    - HuggingFace transformers integration
    - Thread-safe initialization
    
    Usage:
        brain = LocalBrainService()
        response = await brain.think("What is AI?")
    """
    
    _model_instance: Optional[any] = None
    _tokenizer_instance: Optional[any] = None
    _lock: Lock = Lock()
    _initialized: bool = False
    
    def __init__(self):
        """Initialize LocalBrainService."""
        # Thread-safe initialization check
        with self._lock:
            if not self._initialized:
                self._lazy_load_model()
    
    @classmethod
    def _lazy_load_model(cls):
        """
        Lazy load the model and tokenizer (singleton).
        
        Thread-safe lazy initialization.
        """
        if cls._initialized:
            return
        
        with cls._lock:
            # Double-check locking
            if cls._initialized:
                return
            
            logger.info("🧠 Initializing Local Brain Service...")

            # ---------------------------------------------------------
            # 1. Check for Remote API (Client Mode)
            # ---------------------------------------------------------
            try:
                import requests
                from ..core.settings import settings
                
                health_url = f"http://{settings.api_host}:{settings.api_port}/health"
                try:
                    resp = requests.get(health_url, timeout=2)
                    if resp.status_code == 200:
                        logger.info(f"✅ Found running API at {health_url}")
                        logger.info("🚀 Switching to CLIENT MODE (Offloading inference to API)")
                        cls.remote_mode = True
                        cls._initialized = True
                        return
                except requests.exceptions.ConnectionError:
                    logger.info("⚠️ API not reachable, loading local model...")
            except Exception as e:
                logger.warning(f"⚠️ Failed to check API status: {e}")

            # ---------------------------------------------------------
            # 2. Load Local Model (Server Mode)
            # ---------------------------------------------------------
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from ..core.settings import settings
                
                model_path = Path(settings.local_model_path)
                
                # Check if model exists locally
                if model_path.exists() and (model_path / "config.json").exists():
                    logger.info(f"📦 Loading model from {model_path}")
                    model_name_or_path = str(model_path)
                else:
                    logger.info(f"🔽 Model not found locally, will download: {settings.local_model_name}")
                    model_name_or_path = settings.local_model_name
                
                # Detect device
                device = "cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu"
                logger.info(f"🖥️  Using device: {device}")
                
                if device == "cuda":
                    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
                    logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                cls._tokenizer_instance = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    token=settings.huggingface_token
                )
                
                # Add padding token if missing
                if cls._tokenizer_instance.pad_token is None:
                    cls._tokenizer_instance.pad_token = cls._tokenizer_instance.eos_token
                
                # Load model
                logger.info("Loading model...")
                cls._model_instance = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    token=settings.huggingface_token
                )
                
                if device == "cpu":
                    cls._model_instance = cls._model_instance.to(device)
                
                cls._model_instance.eval()  # Set to evaluation mode
                
                cls._initialized = True
                logger.info("✅ Local Brain Service initialized successfully!")
                logger.info(f"   Model: {model_name_or_path}")
                logger.info(f"   Parameters: {cls._model_instance.num_parameters() / 1e6:.1f}M")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Local Brain Service: {e}", exc_info=True)
                raise RuntimeError(f"Local Brain initialization failed: {e}")
    
    async def think(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate a response using the local LLM or Remote API.
        """
        # Check for remote API first (Client Mode)
        if getattr(self, 'remote_mode', False):
            try:
                import requests
                from ..core.settings import settings
                
                api_url = f"http://{settings.api_host}:{settings.api_port}/v4.1/ask"
                payload = {
                    "question": prompt,
                    "use_intent_routing": False,  # Direct thinking
                    "use_web_search": False,
                    "track_experience": False
                }
                
                response = requests.post(api_url, json=payload, timeout=60)
                if response.status_code == 200:
                    return response.json().get("answer", "")
                else:
                    logger.warning(f"⚠️ Remote API failed ({response.status_code}), falling back to local...")
            except Exception as e:
                logger.warning(f"⚠️ Remote API error: {e}, falling back to local...")

        # Fallback to local inference
        if not self._initialized:
            self._lazy_load_model()
        
        from ..core.settings import settings
        
        max_new_tokens = max_tokens or settings.max_tokens
        
        try:
            # Prepare input
            inputs = self._tokenizer_instance(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to same device as model
            device = next(self._model_instance.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self._model_instance.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=settings.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self._tokenizer_instance.pad_token_id,
                    eos_token_id=self._tokenizer_instance.eos_token_id,
                )
            
            # Decode response
            response = self._tokenizer_instance.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the prompt from response (model repeats it)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Error in think(): {e}", exc_info=True)
            return f"Error generating response: {str(e)}"
    
    @classmethod
    def get_model_info(cls) -> dict:
        """Get information about the loaded model."""
        if not cls._initialized:
            return {"status": "not_initialized"}
        
        device = next(cls._model_instance.parameters()).device
        
        return {
            "status": "ready",
            "device": str(device),
            "parameters": cls._model_instance.num_parameters(),
            "parameters_millions": f"{cls._model_instance.num_parameters() / 1e6:.1f}M",
            "model_type": cls._model_instance.config.model_type if hasattr(cls._model_instance.config, 'model_type') else "unknown",
        }
    
    @classmethod
    def reset(cls):
        """Reset the model (useful for testing or reloading)."""
        with cls._lock:
            if cls._model_instance is not None:
                del cls._model_instance
                del cls._tokenizer_instance
                cls._model_instance = None
                cls._tokenizer_instance = None
                cls._initialized = False
                
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("🔄 Local Brain Service reset")


# ============================================================================
# Exports
# ============================================================================

__all__ = ["LocalBrainService"]
