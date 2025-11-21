#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤗 HuggingFace Hub Client - عميل مركز هاجينج فيس
Manages model downloads, uploads, and interactions with HuggingFace Hub
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

# Try to import HuggingFace libraries (optional dependencies)
HAS_HUGGINGFACE = False
HAS_TRANSFORMERS = False

try:
    from huggingface_hub import HfApi, hf_hub_download, list_models, snapshot_download
    HAS_HUGGINGFACE = True
except ImportError:
    logger.warning("⚠️ huggingface_hub not installed. Install with: pip install huggingface_hub")

try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("⚠️ transformers not installed. Install with: pip install transformers torch")


class HFHubClient:
    """
    HuggingFace Hub Client for model management.

    Features:
    - Download models from HuggingFace Hub
    - Upload models to HuggingFace Hub
    - List and search models
    - Load models with quantization support
    - Manage authentication tokens
    """

    def __init__(self, token: Optional[str] = None, org: Optional[str] = None, verbose: bool = False):
        """
        Initialize HuggingFace Hub Client.

        Args:
            token: HuggingFace API token (optional, can use HF_TOKEN env var)
            org: Default organization name
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.org = org

        # Get token from parameter or environment
        self.token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

        # Initialize API client if available
        self.api = None
        if HAS_HUGGINGFACE:
            try:
                self.api = HfApi(token=self.token)
                if self.verbose:
                    logger.info("✅ HuggingFace Hub API initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize HF API: {e}")
        else:
            logger.warning("⚠️ HuggingFace libraries not available - client running in stub mode")

        # Cache directory
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model(
        self,
        repo_id: str,
        revision: str = "main",
        cache_dir: Optional[str] = None
    ) -> str:
        """
        Download a model from HuggingFace Hub.

        Args:
            repo_id: Repository ID (e.g., "bert-base-uncased")
            revision: Git revision (branch, tag, or commit)
            cache_dir: Custom cache directory

        Returns:
            Path to downloaded model directory
        """
        if not HAS_HUGGINGFACE:
            logger.warning(f"⚠️ Cannot download {repo_id} - huggingface_hub not installed")
            # Return a stub path
            stub_path = self.cache_dir / repo_id.replace('/', '_')
            stub_path.mkdir(parents=True, exist_ok=True)
            return str(stub_path)

        try:
            cache = cache_dir or str(self.cache_dir)

            logger.info(f"📥 Downloading model: {repo_id} (revision: {revision})")

            model_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache,
                token=self.token
            )

            logger.info(f"✅ Model downloaded: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"❌ Failed to download {repo_id}: {e}")
            # Return stub path as fallback
            stub_path = self.cache_dir / repo_id.replace('/', '_')
            stub_path.mkdir(parents=True, exist_ok=True)
            return str(stub_path)

    def make_quant_4bit(self) -> Optional[Any]:
        """
        Create 4-bit quantization configuration for memory-efficient inference.

        Returns:
            BitsAndBytesConfig or None if not available
        """
        if not HAS_TRANSFORMERS:
            logger.warning("⚠️ Transformers not available - cannot create quantization config")
            return None

        try:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("✅ Created 4-bit quantization config")
            return config

        except Exception as e:
            logger.error(f"❌ Failed to create quantization config: {e}")
            return None

    def load_text_model(
        self,
        repo_id: str,
        device: str = "cpu",
        quantization: Optional[Any] = None,
        torch_dtype: Optional[Any] = None
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load a text model and tokenizer from HuggingFace.

        Args:
            repo_id: Model repository ID
            device: Device to load on ("cpu", "cuda", "mps")
            quantization: BitsAndBytesConfig for quantization
            torch_dtype: PyTorch dtype (e.g., torch.float16)

        Returns:
            Tuple of (tokenizer, model) or (None, None) if failed
        """
        if not HAS_TRANSFORMERS:
            logger.warning(f"⚠️ Cannot load {repo_id} - transformers not installed")
            return (None, None)

        try:
            logger.info(f"📥 Loading text model: {repo_id}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                token=self.token,
                trust_remote_code=True
            )
            logger.info(f"✅ Tokenizer loaded")

            # Prepare model kwargs
            model_kwargs = {
                'pretrained_model_name_or_path': repo_id,
                'token': self.token,
                'trust_remote_code': True
            }

            # Add quantization config if provided
            if quantization is not None:
                model_kwargs['quantization_config'] = quantization
                model_kwargs['device_map'] = 'auto'
            else:
                model_kwargs['device_map'] = device

            # Add dtype if provided
            if torch_dtype is not None:
                model_kwargs['torch_dtype'] = torch_dtype

            # Load model
            model = AutoModel.from_pretrained(**model_kwargs)
            logger.info(f"✅ Model loaded on {device}")

            return (tokenizer, model)

        except Exception as e:
            logger.error(f"❌ Failed to load model {repo_id}: {e}", exc_info=True)
            return (None, None)

    def push_model_folder(
        self,
        local_dir: str,
        repo_id: str,
        private: bool = False,
        commit_msg: str = "Upload model"
    ) -> Optional[str]:
        """
        Push a local model folder to HuggingFace Hub.

        Args:
            local_dir: Path to local model directory
            repo_id: Target repository ID
            private: Make repository private
            commit_msg: Commit message

        Returns:
            Repository URL or None if failed
        """
        if not HAS_HUGGINGFACE or self.api is None:
            logger.warning(f"⚠️ Cannot push to {repo_id} - HuggingFace API not available")
            return None

        try:
            logger.info(f"📤 Pushing model to {repo_id}")

            # Create repo if it doesn't exist
            try:
                self.api.create_repo(
                    repo_id=repo_id,
                    private=private,
                    exist_ok=True
                )
            except Exception as e:
                logger.debug(f"Repo might already exist: {e}")

            # Upload folder
            self.api.upload_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                commit_message=commit_msg
            )

            repo_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"✅ Model pushed successfully: {repo_url}")
            return repo_url

        except Exception as e:
            logger.error(f"❌ Failed to push model: {e}", exc_info=True)
            return None

    def list_models(
        self,
        query: Optional[str] = None,
        limit: int = 10,
        filter_tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List models from HuggingFace Hub.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_tag: Filter by tag (e.g., "text-generation")

        Returns:
            List of model info dictionaries
        """
        if not HAS_HUGGINGFACE:
            logger.warning("⚠️ Cannot list models - huggingface_hub not installed")
            return []

        try:
            logger.info(f"🔍 Searching models: query={query}, limit={limit}")

            models = list_models(
                search=query,
                limit=limit,
                filter=filter_tag,
                token=self.token
            )

            results = []
            for model in models:
                results.append({
                    'id': model.id,
                    'author': model.author,
                    'downloads': getattr(model, 'downloads', 0),
                    'likes': getattr(model, 'likes', 0),
                    'tags': getattr(model, 'tags', []),
                    'pipeline_tag': getattr(model, 'pipeline_tag', None)
                })

            logger.info(f"✅ Found {len(results)} models")
            return results

        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return []

    def get_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            repo_id: Repository ID

        Returns:
            Model info dictionary or None
        """
        if not HAS_HUGGINGFACE or self.api is None:
            logger.warning(f"⚠️ Cannot get info for {repo_id} - API not available")
            return None

        try:
            info = self.api.model_info(repo_id=repo_id, token=self.token)

            return {
                'id': info.id,
                'author': info.author,
                'sha': info.sha,
                'created_at': str(info.created_at),
                'last_modified': str(info.last_modified),
                'private': info.private,
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags,
                'pipeline_tag': info.pipeline_tag,
                'library_name': getattr(info, 'library_name', None)
            }

        except Exception as e:
            logger.error(f"❌ Failed to get model info: {e}")
            return None

    def is_available(self) -> bool:
        """Check if HuggingFace libraries are available."""
        return HAS_HUGGINGFACE and HAS_TRANSFORMERS


# Convenience functions
def download_model(repo_id: str, revision: str = "main") -> str:
    """Quick function to download a model."""
    client = HFHubClient()
    return client.download_model(repo_id, revision)


def load_model(repo_id: str, device: str = "cpu", use_4bit: bool = False):
    """Quick function to load a model."""
    client = HFHubClient()
    quant_config = client.make_quant_4bit() if use_4bit else None
    return client.load_text_model(repo_id, device, quantization=quant_config)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Testing HFHubClient...")
    client = HFHubClient(verbose=True)

    print(f"\n✅ Client available: {client.is_available()}")
    print(f"📂 Cache directory: {client.cache_dir}")

    if client.is_available():
        # Test model search
        models = client.list_models(query="bert", limit=3)
        print(f"\n🔍 Found {len(models)} BERT models")
        for model in models:
            print(f"   - {model['id']} ({model['downloads']} downloads)")
    else:
        print("\n⚠️ HuggingFace libraries not installed - client in stub mode")
        print("   Install with: pip install huggingface_hub transformers torch")
