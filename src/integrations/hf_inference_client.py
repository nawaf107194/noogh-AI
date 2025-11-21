#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 HuggingFace Inference Client - عميل الاستنتاج من هاجينج فيس
Provides inference capabilities using HuggingFace Inference API
"""

import logging
import os
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

# Try to import HuggingFace Inference API (optional dependency)
HAS_INFERENCE_API = False

try:
    from huggingface_hub import InferenceClient as HFInferenceAPI
    HAS_INFERENCE_API = True
except ImportError:
    logger.warning("⚠️ huggingface_hub not installed. Install with: pip install huggingface_hub")


class HFInferenceClient:
    """
    HuggingFace Inference Client for running models via API.

    Features:
    - Text generation
    - Text classification
    - Embeddings generation
    - Question answering
    - Summarization
    - Translation
    - Statistics tracking
    """

    def __init__(self, token: Optional[str] = None, verbose: bool = False):
        """
        Initialize HuggingFace Inference Client.

        Args:
            token: HuggingFace API token (optional, can use HF_TOKEN env var)
            verbose: Enable verbose logging
        """
        self.verbose = verbose

        # Get token from parameter or environment
        self.token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

        # Initialize API client if available
        self.client = None
        if HAS_INFERENCE_API:
            try:
                self.client = HFInferenceAPI(token=self.token)
                if self.verbose:
                    logger.info("✅ HuggingFace Inference API initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Inference API: {e}")
        else:
            logger.warning("⚠️ HuggingFace Inference API not available - client running in stub mode")

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

    def generate(
        self,
        model: str,
        inputs: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate text using a language model.

        Args:
            model: Model name/ID
            inputs: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters

        Returns:
            Dictionary with generated_text key
        """
        self.total_calls += 1

        if not HAS_INFERENCE_API or self.client is None:
            logger.warning(f"⚠️ Cannot generate - Inference API not available")
            self.failed_calls += 1
            return {"generated_text": f"[Stub] Generated text for: {inputs[:50]}..."}

        try:
            logger.info(f"🔮 Generating text with {model}")

            result = self.client.text_generation(
                prompt=inputs,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                return_full_text=False,
                **kwargs
            )

            self.successful_calls += 1
            logger.info(f"✅ Generation successful")

            return {"generated_text": result}

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"❌ Generation failed: {e}")
            return {"generated_text": f"[Error] {str(e)}"}

    def classify(
        self,
        model: str,
        inputs: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify text using a classification model.

        Args:
            model: Model name/ID
            inputs: Text to classify
            top_k: Number of top predictions to return

        Returns:
            List of predictions with label and score
        """
        self.total_calls += 1

        if not HAS_INFERENCE_API or self.client is None:
            logger.warning(f"⚠️ Cannot classify - Inference API not available")
            self.failed_calls += 1
            return [{"label": "STUB_LABEL", "score": 0.9}]

        try:
            logger.info(f"🏷️ Classifying text with {model}")

            result = self.client.text_classification(
                text=inputs,
                model=model,
                top_k=top_k
            )

            self.successful_calls += 1
            logger.info(f"✅ Classification successful: {len(result)} predictions")

            return result

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"❌ Classification failed: {e}")
            return [{"label": "ERROR", "score": 0.0, "error": str(e)}]

    def embed(
        self,
        model: str,
        inputs: Union[str, List[str]]
    ) -> List[List[float]]:
        """
        Generate embeddings for text.

        Args:
            model: Embedding model name/ID
            inputs: Text or list of texts to embed

        Returns:
            List of embedding vectors
        """
        self.total_calls += 1

        if not HAS_INFERENCE_API or self.client is None:
            logger.warning(f"⚠️ Cannot embed - Inference API not available")
            self.failed_calls += 1
            # Return dummy embeddings
            if isinstance(inputs, str):
                return [[0.1] * 384]  # Common embedding dimension
            return [[0.1] * 384 for _ in inputs]

        try:
            logger.info(f"🔢 Generating embeddings with {model}")

            result = self.client.feature_extraction(
                text=inputs,
                model=model
            )

            self.successful_calls += 1
            logger.info(f"✅ Embedding successful")

            return result

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"❌ Embedding failed: {e}")
            # Return dummy embeddings
            if isinstance(inputs, str):
                return [[0.0] * 384]
            return [[0.0] * 384 for _ in inputs]

    def question_answer(
        self,
        model: str,
        question: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Answer a question based on context.

        Args:
            model: QA model name/ID
            question: Question to answer
            context: Context containing the answer

        Returns:
            Dictionary with answer, score, start, and end
        """
        self.total_calls += 1

        if not HAS_INFERENCE_API or self.client is None:
            logger.warning(f"⚠️ Cannot answer question - Inference API not available")
            self.failed_calls += 1
            return {
                "answer": "[Stub] Answer not available",
                "score": 0.0,
                "start": 0,
                "end": 0
            }

        try:
            logger.info(f"❓ Answering question with {model}")

            result = self.client.question_answering(
                question=question,
                context=context,
                model=model
            )

            self.successful_calls += 1
            logger.info(f"✅ Question answered: {result['answer']}")

            return result

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"❌ Question answering failed: {e}")
            return {
                "answer": f"[Error] {str(e)}",
                "score": 0.0,
                "start": 0,
                "end": 0
            }

    def summarize(
        self,
        model: str,
        inputs: str,
        max_length: int = 130,
        min_length: int = 30
    ) -> Dict[str, str]:
        """
        Summarize text.

        Args:
            model: Summarization model name/ID
            inputs: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Dictionary with summary_text key
        """
        self.total_calls += 1

        if not HAS_INFERENCE_API or self.client is None:
            logger.warning(f"⚠️ Cannot summarize - Inference API not available")
            self.failed_calls += 1
            return {"summary_text": "[Stub] Summary not available"}

        try:
            logger.info(f"📝 Summarizing text with {model}")

            result = self.client.summarization(
                text=inputs,
                model=model,
                max_length=max_length,
                min_length=min_length
            )

            self.successful_calls += 1
            logger.info(f"✅ Summarization successful")

            return {"summary_text": result['summary_text']}

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"❌ Summarization failed: {e}")
            return {"summary_text": f"[Error] {str(e)}"}

    def translate(
        self,
        model: str,
        inputs: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Translate text.

        Args:
            model: Translation model name/ID
            inputs: Text to translate
            src_lang: Source language code (optional)
            tgt_lang: Target language code (optional)

        Returns:
            Dictionary with translation_text key
        """
        self.total_calls += 1

        if not HAS_INFERENCE_API or self.client is None:
            logger.warning(f"⚠️ Cannot translate - Inference API not available")
            self.failed_calls += 1
            return {"translation_text": "[Stub] Translation not available"}

        try:
            logger.info(f"🌐 Translating text with {model}")

            # Build parameters
            params = {}
            if src_lang:
                params['src_lang'] = src_lang
            if tgt_lang:
                params['tgt_lang'] = tgt_lang

            result = self.client.translation(
                text=inputs,
                model=model,
                **params
            )

            self.successful_calls += 1
            logger.info(f"✅ Translation successful")

            return {"translation_text": result['translation_text']}

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"❌ Translation failed: {e}")
            return {"translation_text": f"[Error] {str(e)}"}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dictionary with call statistics
        """
        success_rate = (
            self.successful_calls / self.total_calls
            if self.total_calls > 0
            else 1.0
        )

        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "api_available": HAS_INFERENCE_API and self.client is not None
        }

    def is_available(self) -> bool:
        """Check if Inference API is available."""
        return HAS_INFERENCE_API and self.client is not None


# Convenience functions
def generate_text(model: str, prompt: str, max_tokens: int = 50) -> str:
    """Quick function to generate text."""
    client = HFInferenceClient()
    result = client.generate(model, prompt, max_new_tokens=max_tokens)
    return result.get("generated_text", "")


def classify_text(model: str, text: str, top_k: int = 5) -> List[Dict]:
    """Quick function to classify text."""
    client = HFInferenceClient()
    return client.classify(model, text, top_k)


def get_embeddings(model: str, texts: Union[str, List[str]]) -> List[List[float]]:
    """Quick function to get embeddings."""
    client = HFInferenceClient()
    return client.embed(model, texts)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Testing HFInferenceClient...")
    client = HFInferenceClient(verbose=True)

    print(f"\n✅ Client available: {client.is_available()}")

    if client.is_available():
        # Test text generation
        print("\n🔮 Testing text generation...")
        result = client.generate(
            model="gpt2",
            inputs="The future of AI is",
            max_new_tokens=20
        )
        print(f"   Generated: {result['generated_text']}")

        # Test classification
        print("\n🏷️ Testing classification...")
        result = client.classify(
            model="distilbert-base-uncased-finetuned-sst-2-english",
            inputs="I love this!",
            top_k=2
        )
        print(f"   Predictions: {result}")

    else:
        print("\n⚠️ HuggingFace Inference API not available - client in stub mode")
        print("   Install with: pip install huggingface_hub")

    # Show stats
    stats = client.get_stats()
    print(f"\n📊 Stats: {stats}")
