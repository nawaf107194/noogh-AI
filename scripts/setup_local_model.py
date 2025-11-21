#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Model Setup Script
=========================

Downloads a lightweight local LLM for testing the Sovereign AI pipeline.
"""

import asyncio
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def download_local_model():
    """Download a small local model for testing."""
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.core.settings import settings
        
        logger.info("=" * 70)
        logger.info("🤖 Noogh Local Model Setup")
        logger.info("=" * 70)
        
        # Create models directory
        models_dir = Path(settings.local_model_path)
        models_dir.parent.mkdir(parents=True, exist_ok=True)
        
        model_name = settings.local_model_name
        logger.info(f"\n📦 Downloading model: {model_name}")
        logger.info(f"   Destination: {models_dir}")
        
        # Check if already exists
        if models_dir.exists() and (models_dir / "config.json").exists():
            logger.info("✅ Model already exists locally!")
            logger.info("\nTo re-download, delete the models directory:")
            logger.info(f"   rm -rf {models_dir}")
            return
        
        logger.info("\n🔽 Starting download (this may take a few minutes)...")
        
        # Download tokenizer
        logger.info("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=settings.huggingface_token
        )
        
        # Download model
        logger.info("   Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=settings.huggingface_token
        )
        
        # Save locally
        logger.info(f"\n💾 Saving model to {models_dir}...")
        tokenizer.save_pretrained(models_dir)
        model.save_pretrained(models_dir)
        
        logger.info("\n✅ Model download complete!")
        logger.info("\n" + "=" * 70)
        logger.info("Model Information:")
        logger.info("=" * 70)
        logger.info(f"Model Name: {model_name}")
        logger.info(f"Save Path: {models_dir}")
        logger.info(f"Parameters: {model.num_parameters() / 1e6:.1f}M")
        logger.info(f"Model Type: {model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'}")
        logger.info(f"Vocab Size: {model.config.vocab_size if hasattr(model.config, 'vocab_size') else 'unknown'}")
        logger.info("=" * 70)
        
        logger.info("\n🧪 Testing model...")
        test_input = "Hello, I am Noogh, an AI assistant."
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"   Test input: {test_input}")
        logger.info(f"   Test output: {response}")
        
        logger.info("\n🎉 Setup complete! You can now run the server:")
        logger.info("   python -m src.api.main")
        logger.info("\n💡 Your AI is now 100% local and sovereign! 🇸🇦")
        
    except ImportError as e:
        logger.error("❌ Missing dependencies! Install them first:")
        logger.error("   pip install transformers torch accelerate sentencepiece")
        logger.error(f"\nError: {e}")
    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}", exc_info=True)


async def list_available_models():
    """Show recommended models for local inference."""
    
    print("\n" + "=" * 70)
    print("📚 Recommended Local Models for Noogh")
    print("=" * 70)
    
    models = [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B",
            "memory": "~2.5 GB",
            "speed": "Very Fast",
            "quality": "Good for testing",
            "recommended": True
        },
        {
            "name": "microsoft/phi-2",
            "size": "2.7B",
            "memory": "~6 GB",
            "speed": "Fast",
            "quality": "Excellent",
            "recommended": True
        },
        {
            "name": "stabilityai/stablelm-2-1_6b",
            "size": "1.6B",
            "memory": "~4 GB",
            "speed": "Fast",
            "quality": "Good",
            "recommended": False
        },
        {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "size": "7B",
            "memory": "~14 GB",
            "speed": "Medium",
            "quality": "Excellent",
            "recommended": False
        },
    ]
    
    for model in models:
        star = "⭐ " if model["recommended"] else "   "
        print(f"\n{star}{model['name']}")
        print(f"   Size: {model['size']} | Memory: {model['memory']} | Speed: {model['speed']}")
        print(f"   Quality: {model['quality']}")
    
    print("\n" + "=" * 70)
    print("\n💡 To use a different model, set in .env:")
    print("   LOCAL_MODEL_NAME=microsoft/phi-2")
    print("\n⚠️  Note: Larger models require more RAM/VRAM")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    import torch
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        asyncio.run(list_available_models())
    else:
        asyncio.run(download_local_model())
