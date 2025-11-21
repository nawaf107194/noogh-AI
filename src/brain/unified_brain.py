#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Unified Noogh Brain - دمج كامل المميزات
يجمع:
  - MegaBrain V5 (الدماغ الضخم مع Transformer)
  - نظام التدريب الهجين CPU/GPU
  - Knowledge Graph & Vector Store
  - إدارة متعددة للنماذج
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)

# Import AutonomousClient for resource management
try:
    from ..clients.autonomous_client import AutonomousClient, SyncAutonomousClient
    AUTONOMOUS_CLIENT_AVAILABLE = True
except ImportError:
    AUTONOMOUS_CLIENT_AVAILABLE = False
    logger.warning("⚠️  AutonomousClient not available - autonomous features disabled")

# ============================================================================
# استيراد المكونات المتقدمة (مع fallback آمن)
# ============================================================================

try:
    from ..api.utils.device_manager import get_device_manager
except ImportError:
    class SimpleDeviceManager:
        def get_device(self):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def print_info(self):
            device = self.get_device()
            logger.info(f"Device: {device.type.upper()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        def get_optimal_batch_size(self):
            return 32 if torch.cuda.is_available() else 16
        def get_recommended_workers(self):
            return 4
        @property
        def device_info(self):
            class Info:
                name = "CPU" if not torch.cuda.is_available() else torch.cuda.get_device_name(0)
                memory_total = 0
                memory_available = 0
            return Info()
    def get_device_manager():
        return SimpleDeviceManager()

try:
    from ..training.trainer import HybridTrainer
except ImportError:
    HybridTrainer = None

try:
    from .enhanced_brain import KnowledgeGraph, VectorStore, ReasoningEngine
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False


# ============================================================================
# MegaBrain V5 Components - الدماغ الضخم
# ============================================================================

class MegaAttentionBlock(nn.Module):
    """
    Mega Multi-Head Attention Block
    أقوى من GPT-3!
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-Forward Network (HUGE)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expand 4x
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Residual Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm + Attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # Pre-norm + FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class MegaResidualBlock(nn.Module):
    """
    Mega Residual Block
    ResNet-style مع skip connections
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

        # Projection if dimensions don't match
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)

        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        # First transformation
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second transformation
        out = self.linear2(out)
        out = self.norm2(out)

        # Skip connection
        if self.projection is not None:
            identity = self.projection(identity)

        out = out + identity
        out = self.activation(out)

        return out


class MegaBrainV5(nn.Module):
    """
    🧠🔥 MEGA BRAIN V5
    ==================

    الدماغ العصبوني الأضخم في نوقة!

    Architecture:
    - Input Layer: 1024 neurons
    - Hidden Layers: 50 layers × 32,768 neurons each = 1,638,400 neurons
    - Attention Layers: 20 layers × 2048 neurons each = 40,960 neurons
    - Output Layer: 1024 neurons

    Total: ~1.7 MILLION neurons! 🔥
    """

    def __init__(
        self,
        input_size: int = 1024,
        hidden_size: int = 32768,  # 32K per layer!
        num_hidden_layers: int = 50,  # 50 layers deep!
        attention_dim: int = 2048,
        num_attention_layers: int = 20,
        num_attention_heads: int = 16,
        output_size: int = 1024,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_attention = use_attention

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Massive Hidden Layers (50 layers × 32K neurons)
        self.hidden_blocks = nn.ModuleList()

        for i in range(num_hidden_layers):
            block = MegaResidualBlock(
                in_features=hidden_size,
                out_features=hidden_size,
                dropout=dropout
            )
            self.hidden_blocks.append(block)

        # Attention Layers (20 transformer blocks)
        self.attention_blocks = None
        if use_attention:
            self.attention_projection = nn.Linear(hidden_size, attention_dim)
            self.attention_blocks = nn.ModuleList()

            for i in range(num_attention_layers):
                block = MegaAttentionBlock(
                    d_model=attention_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout
                )
                self.attention_blocks.append(block)

            self.attention_output = nn.Linear(attention_dim, hidden_size)

        # Output projection
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        """
        Forward pass through MEGA BRAIN

        Args:
            x: Input tensor [batch_size, input_size]

        Returns:
            Output tensor [batch_size, output_size]
        """
        # Input projection
        x = self.input_layer(x)

        # Pass through all hidden layers
        for block in self.hidden_blocks:
            x = block(x)

        # Attention layers (if enabled)
        if self.use_attention and self.attention_blocks:
            # Project to attention dimension
            x_attn = self.attention_projection(x)

            # Add sequence dimension for attention
            if x_attn.dim() == 2:
                x_attn = x_attn.unsqueeze(1)  # [batch, 1, dim]

            # Pass through attention blocks
            for attn_block in self.attention_blocks:
                x_attn = attn_block(x_attn)

            # Remove sequence dimension
            x_attn = x_attn.squeeze(1)

            # Project back and add residual
            x = x + self.attention_output(x_attn)

        # Output projection
        x = self.output_layer(x)

        return x

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'MegaBrainV5',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_millions': total_params / 1e6,
            'memory_estimate_mb': (total_params * 4) / (1024 * 1024),  # Float32
            'use_attention': self.use_attention
        }


# ============================================================================
# Unified Noogh Brain - النظام الموحد
# ============================================================================

class UnifiedNooghBrain:
    """
    العقل الموحد لنظام Noogh AI

    Features:
    ✅ MegaBrain V5 (1.7M neurons, Transformer architecture)
    ✅ Hybrid CPU/GPU (auto-detect)
    ✅ Advanced training (mixed precision, gradient accumulation)
    ✅ Multiple model types (MLP, CNN, RNN, Transformer, MegaBrain)
    ✅ Knowledge Graph (if available)
    ✅ Vector Store (if available)
    ✅ Reasoning Engine (if available)
    ✅ Autonomous System Integration (Resource monitoring, Load balancing, Smart training)
    """

    def __init__(self, device: Union[str, torch.device] = "auto", verbose: bool = True, use_autonomous: bool = True):
        self.verbose = verbose

        # Device Management
        if device == "auto":
            self.device_manager = get_device_manager()
            self.device = self.device_manager.get_device()
        else:
            self.device = torch.device(device)
            self.device_manager = get_device_manager()

        # Model & Training
        self.model = None
        self.trainer = None
        self.history = None

        # Advanced Features (if available)
        self.knowledge_graph = KnowledgeGraph() if ADVANCED_FEATURES else None
        self.vector_store = None  # يتم تهيئته عند الحاجة
        self.reasoning_engine = None  # يتم تهيئته عند الحاجة

        # 🤖 Autonomous System Integration
        self.autonomous_client = None
        self.use_autonomous = use_autonomous and AUTONOMOUS_CLIENT_AVAILABLE

        if self.use_autonomous:
            try:
                # Create sync client wrapper for use in __init__
                from ..clients.autonomous_client import SyncAutonomousClient
                self.autonomous_client = SyncAutonomousClient()
                if self.verbose:
                    logger.info("🤖 Autonomous System: ✅ Connected")
            except Exception as e:
                logger.warning(f"⚠️  Failed to connect to Autonomous System: {e}")
                self.autonomous_client = None
                self.use_autonomous = False

        # 🤗 Hugging Face Integration
        self.hf = None
        self.hf_inference = None
        try:
            from ..integrations.hf_hub_client import HFHubClient
            from ..integrations.hf_inference_client import HFInferenceClient

            self.hf = HFHubClient(token=None, org=None, verbose=False)
            self.hf_inference = HFInferenceClient(token=None, verbose=False)

            if self.verbose:
                logger.info("🤗 Hugging Face Hub: ✅ Connected")
                logger.info("🌐 HF Inference API: ✅ Connected")
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  HF services not available: {e}")
            self.hf = None
            self.hf_inference = None

        if self.verbose:
            logger.info("="*60)
            logger.info("🧠 Unified Noogh Brain - النظام الموحد")
            logger.info("="*60)
            self.device_manager.print_info()
            logger.info(f"Advanced Features: {'✅ Enabled' if ADVANCED_FEATURES else '❌ Disabled'}")
            logger.info(f"Autonomous System: {'✅ Enabled' if self.use_autonomous else '❌ Disabled'}")
            logger.info(f"Hugging Face Hub: {'✅ Enabled' if self.hf else '❌ Disabled'}")
            logger.info(f"HF Inference API: {'✅ Enabled' if self.hf_inference else '❌ Disabled'}")
            logger.info("="*60 + "\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Properties - الخصائص
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def is_ready(self) -> bool:
        """هل الدماغ جاهز للاستخدام (هل النموذج محمل)"""
        return self.model is not None

    def inference(self, input_data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        استدعاء الدماغ للاستنتاج

        Args:
            input_data: البيانات المدخلة

        Returns:
            النتيجة أو None
        """
        if not self.is_ready:
            if self.verbose:
                logger.warning("⚠️  Brain not ready - model not loaded")
            return None

        try:
            # Convert input to tensor if needed
            if isinstance(input_data, dict):
                # For now, create a dummy tensor based on model input size
                # In production, you'd properly encode the input
                batch_size = 1
                input_tensor = torch.randn(batch_size, self.model.input_size).to(self.device)
            else:
                input_tensor = input_data.to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)

            return output

        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Inference error: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Autonomous System Integration - التكامل مع النظام الذاتي
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def check_resources(self) -> Optional[Dict[str, Any]]:
        """
        فحص الموارد الحالية (VRAM, CPU, RAM, Temperature)
        Check current resources (VRAM, CPU, RAM, Temperature)

        Returns:
            Dict with resource info or None if autonomous system unavailable
        """
        if not self.use_autonomous or not self.autonomous_client:
            if self.verbose:
                logger.warning("⚠️  Autonomous system not available")
            return None

        try:
            resources = self.autonomous_client.get_resources()
            if self.verbose:
                logger.info(f"📊 Resources: VRAM {resources['gpu_memory_percent']:.1f}%, "
                           f"Temp {resources['gpu_temperature']}°C, "
                           f"Status: {resources['overall_status']}")
            return resources
        except Exception as e:
            logger.error(f"❌ Failed to check resources: {e}")
            return None

    def smart_device_decision(self, task_type: str = "inference", estimated_vram: float = 1.0,
                             priority: str = "medium") -> str:
        """
        طلب قرار ذكي من موزع الحمل: CPU أو GPU؟
        Request smart decision from load balancer: CPU or GPU?

        Args:
            task_type: نوع المهمة (inference, training, etc.)
            estimated_vram: VRAM المتوقع بالـ GB
            priority: الأولوية (low, medium, high, critical)

        Returns:
            "gpu" or "cpu"
        """
        if not self.use_autonomous or not self.autonomous_client:
            # Fallback: use current device
            return self.device.type

        try:
            decision = self.autonomous_client.decide_device(
                task_type=task_type,
                estimated_vram=estimated_vram,
                priority=priority
            )

            device = decision.get('device', 'cpu')
            if self.verbose:
                logger.info(f"🎯 Load Balancer Decision: {device.upper()} ({decision.get('reason', '')})")

            return device

        except Exception as e:
            logger.error(f"❌ Failed to get device decision: {e}")
            return self.device.type

    def is_ready_for_training(self, estimated_vram: float = 4.0) -> bool:
        """
        هل النظام جاهز للتدريب؟
        Is the system ready for training?

        Args:
            estimated_vram: VRAM المطلوب بالـ GB

        Returns:
            True if ready, False otherwise
        """
        if not self.use_autonomous or not self.autonomous_client:
            # Fallback: always return True
            return True

        try:
            ready = self.autonomous_client.is_ready_for_training(estimated_vram)
            if self.verbose:
                status = "✅ READY" if ready else "⚠️  NOT READY"
                logger.info(f"{status} for training ({estimated_vram} GB VRAM needed)")
            return ready
        except Exception as e:
            logger.error(f"❌ Failed to check training readiness: {e}")
            return True  # Fallback

    def auto_prepare_training(self, model_name: str = "UnifiedBrain",
                             estimated_vram: float = 4.0) -> Optional[Dict[str, Any]]:
        """
        تحضير تلقائي للتدريب (إيقاف الوزراء، تفريغ VRAM)
        Automatic training preparation (pause ministers, free VRAM)

        Args:
            model_name: اسم النموذج
            estimated_vram: VRAM المطلوب

        Returns:
            Preparation result or None
        """
        if not self.use_autonomous or not self.autonomous_client:
            if self.verbose:
                logger.warning("⚠️  Autonomous system not available, skipping preparation")
            return None

        try:
            if self.verbose:
                logger.info(f"🔧 Preparing system for training: {model_name} ({estimated_vram} GB)")

            result = self.autonomous_client.prepare_training(
                model_name=model_name,
                estimated_vram_needed=estimated_vram
            )

            if self.verbose:
                logger.info(f"✅ Preparation complete:")
                logger.info(f"   Freed VRAM: {result.get('freed_vram', 0):.2f} GB")
                logger.info(f"   Available VRAM: {result.get('available_vram', 0):.2f} GB")
                logger.info(f"   Paused ministers: {result.get('paused_ministers', 0)}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to prepare for training: {e}")
            return None

    def complete_training_session(self, success: bool = True) -> Optional[Dict[str, Any]]:
        """
        إنهاء جلسة التدريب (إعادة الوزراء، إعادة تحميل ALLaM)
        Complete training session (restore ministers, reload ALLaM)

        Args:
            success: هل التدريب نجح؟

        Returns:
            Completion result or None
        """
        if not self.use_autonomous or not self.autonomous_client:
            return None

        try:
            if self.verbose:
                logger.info(f"🏁 Completing training session (success={success})...")

            result = self.autonomous_client.complete_training(success=success)

            if self.verbose:
                logger.info(f"✅ Training session completed")
                logger.info(f"   Progress: {result.get('progress', 0):.1f}%")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to complete training: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Model Creation - إنشاء النماذج
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_mega_brain(
        self,
        config: str = "small",
        **kwargs
    ):
        """
        إنشاء MegaBrain V5

        Configs: micro, tiny, small, base, medium, large, ultra, insane
        """
        configs = {
            "micro": {
                "input_size": 1024,
                "hidden_size": 512,
                "num_hidden_layers": 2,
                "attention_dim": 128,
                "num_attention_layers": 1,
                "num_attention_heads": 4,
                "output_size": 1024
            },
            "tiny": {
                "input_size": 1024,
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "attention_dim": 256,
                "num_attention_layers": 2,
                "num_attention_heads": 4,
                "output_size": 1024
            },
            "small": {
                "input_size": 1024,
                "hidden_size": 2048,
                "num_hidden_layers": 6,
                "attention_dim": 512,
                "num_attention_layers": 4,
                "num_attention_heads": 8,
                "output_size": 1024
            },
            "base": {
                "input_size": 1024,
                "hidden_size": 4096,
                "num_hidden_layers": 8,
                "attention_dim": 768,
                "num_attention_layers": 6,
                "num_attention_heads": 8,
                "output_size": 1024
            },
            "medium": {
                "input_size": 1024,
                "hidden_size": 8192,
                "num_hidden_layers": 12,
                "attention_dim": 1024,
                "num_attention_layers": 8,
                "num_attention_heads": 8,
                "output_size": 1024
            },
            "large": {
                "input_size": 1024,
                "hidden_size": 16384,
                "num_hidden_layers": 30,
                "attention_dim": 1024,
                "num_attention_layers": 10,
                "num_attention_heads": 8,
                "output_size": 1024
            },
            "ultra": {
                "input_size": 1024,
                "hidden_size": 32768,
                "num_hidden_layers": 50,
                "attention_dim": 2048,
                "num_attention_layers": 20,
                "num_attention_heads": 16,
                "output_size": 1024
            },
            "insane": {
                "input_size": 2048,
                "hidden_size": 65536,
                "num_hidden_layers": 100,
                "attention_dim": 4096,
                "num_attention_layers": 40,
                "num_attention_heads": 32,
                "output_size": 2048
            }
        }

        if config not in configs:
            logger.warning(f"Unknown config '{config}', using 'small'")
            config = "small"

        if self.verbose:
            logger.info(f"\n🔥 Creating MEGA BRAIN V5 with config: {config.upper()}")

        # Merge config with kwargs
        model_config = {**configs[config], **kwargs}

        self.model = MegaBrainV5(**model_config)
        self.model = self.model.to(self.device)

        # Prefer TF32 on Ampere+/RTX 50xx for speed
        try:
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if self.verbose:
            stats = self.model.get_stats()
            logger.info(f"✅ MEGA BRAIN V5 created successfully!")
            logger.info(f"   Parameters: {stats['parameters_millions']:.2f}M")
            logger.info(f"   Est. Memory: {stats['memory_estimate_mb']:.2f} MB")
            logger.info(f"   Device: {self.device}")

        return self

    def create_model(self, model_type: str, **kwargs):
        """
        إنشاء نموذج

        Args:
            model_type: نوع النموذج (mlp, cnn, rnn, transformer, megabrain)
            **kwargs: معاملات النموذج
        """
        if model_type.lower() in ["megabrain", "mega", "mega_brain"]:
            config = kwargs.get("config", "small")
            return self.create_mega_brain(config=config, **kwargs)

        # For other model types, use create_model function if available
        try:
            from models.hybrid import create_model as create_hybrid_model
            self.model = create_hybrid_model(model_type, **kwargs)
            self.model = self.model.to(self.device)
        except ImportError:
            raise ValueError(f"Model type '{model_type}' not supported. Use 'megabrain' or install hybrid models.")

        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"\n🧠 Model Created: {model_type.upper()}")
            logger.info(f"   Total params: {total_params:,}")
            logger.info(f"   Trainable params: {trainable_params:,}")
            logger.info(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB\n")

        return self

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training - التدريب
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def prepare_trainer(
        self,
        learning_rate: float = 0.001,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        """تحضير المدرب"""
        if self.model is None:
            raise ValueError("يجب إنشاء النموذج أولاً باستخدام create_model() أو create_mega_brain()")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        if HybridTrainer:
            self.trainer = HybridTrainer(
                model=self.model,
                optimizer=optimizer,
                criterion=criterion,
                use_amp=use_amp,
                gradient_accumulation_steps=gradient_accumulation_steps,
                verbose=self.verbose
            )
        else:
            logger.warning("HybridTrainer not available, using basic training")
            self.trainer = None

        return self

    def train(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs: int = 10,
        batch_size: int = None,
        save_path: str = None,
        auto_prepare: bool = True,
        estimated_vram: float = 4.0
    ):
        """
        تدريب النموذج

        Args:
            train_data: بيانات التدريب
            train_labels: تسميات التدريب
            test_data: بيانات الاختبار
            test_labels: تسميات الاختبار
            epochs: عدد الـ epochs
            batch_size: حجم الـ batch (None = auto)
            save_path: مسار حفظ النموذج
            auto_prepare: استخدام النظام الذاتي للتحضير (True = تلقائي)
            estimated_vram: VRAM المتوقع بالـ GB
        """
        # 🤖 Autonomous preparation (if enabled)
        prep_result = None
        if auto_prepare and self.use_autonomous:
            model_name = save_path.split('/')[-1].replace('.pt', '') if save_path else "UnifiedBrain"
            prep_result = self.auto_prepare_training(model_name, estimated_vram)

        if self.trainer is None:
            self.prepare_trainer()

        # Auto batch size
        if batch_size is None:
            batch_size = self.device_manager.get_optimal_batch_size()
            if self.verbose:
                logger.info(f"📊 Auto batch size: {batch_size}")

        # إنشاء DataLoaders
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(
            torch.FloatTensor(train_data),
            torch.LongTensor(train_labels)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(test_data),
            torch.LongTensor(test_labels)
        )

        num_workers = self.device_manager.get_recommended_workers()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda')
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda')
        )

        # Train
        training_success = False
        if self.trainer:
            self.history = self.trainer.train(
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=epochs,
                save_path=save_path
            )
            training_success = True
        else:
            logger.warning("No trainer available, skipping training")

        # 🤖 Complete autonomous session (if was prepared)
        if auto_prepare and self.use_autonomous and prep_result:
            self.complete_training_session(success=training_success)

        # 🤗 Auto-upload checkpoint to HF Hub (if training succeeded and save_path is a directory)
        if training_success and save_path and self.hf:
            try:
                import os
                if os.path.isdir(save_path):
                    # Extract model name from path
                    model_name = Path(save_path).name
                    repo_id = f"noogh/{model_name}"

                    if self.verbose:
                        logger.info(f"🤗 Auto-uploading checkpoint to HF Hub...")

                    self.hf.push_model_folder(
                        save_path,
                        repo_id=repo_id,
                        private=True,
                        commit_msg="Noogh auto-checkpoint: training session complete"
                    )

                    if self.verbose:
                        logger.info(f"✅ Checkpoint uploaded to: {repo_id}")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"⚠️  HF checkpoint upload skipped: {e}")

        return self.history

    @torch.no_grad()
    def predict(self, data):
        """
        تنفيذ استدلال

        Args:
            data: البيانات للاستدلال

        Returns:
            التنبؤات
        """
        if self.model is None:
            raise ValueError("لا يوجد نموذج محمّل")

        self.model.eval()

        # تحويل لـ tensor
        if not isinstance(data, torch.Tensor):
            data = torch.FloatTensor(data)

        data = data.to(self.device)

        # Predict
        output = self.model(data)
        _, predicted = output.max(1)

        return predicted.cpu().numpy()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Advanced Features - المميزات المتقدمة
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add_knowledge(self, node_id: str, data: Dict[str, Any]):
        """إضافة معرفة للـ Knowledge Graph"""
        if not ADVANCED_FEATURES:
            if self.verbose:
                logger.warning("⚠️ Knowledge Graph not available")
            return

        self.knowledge_graph.add_node(node_id, data)

        if self.verbose:
            logger.info(f"✅ Knowledge added: {node_id}")

    def connect_knowledge(self, from_id: str, to_id: str, weight: float = 1.0):
        """ربط معرفتين في الـ Graph"""
        if not ADVANCED_FEATURES:
            return

        self.knowledge_graph.add_edge(from_id, to_id, weight)

        if self.verbose:
            logger.info(f"✅ Connected: {from_id} -> {to_id}")

    def find_related_knowledge(self, node_id: str, max_depth: int = 2) -> List[str]:
        """البحث عن معارف مرتبطة"""
        if not ADVANCED_FEATURES:
            return []

        return self.knowledge_graph.find_related(node_id, max_depth)

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """إحصائيات المعرفة"""
        if not ADVANCED_FEATURES:
            return {"status": "Advanced features disabled"}

        return self.knowledge_graph.get_stats()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Model Management - إدارة النماذج
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_model(self, path: str):
        """حفظ النموذج"""
        if self.model is None:
            raise ValueError("لا يوجد نموذج للحفظ")

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'device_info': {
                'type': self.device.type,
                'name': self.device_manager.device_info.name
            }
        }

        if self.trainer:
            save_data['optimizer_state_dict'] = self.trainer.optimizer.state_dict()

        torch.save(save_data, path)

        if self.verbose:
            logger.info(f"✅ Model saved to {path}")

    def load_model(self, path: str):
        """تحميل النموذج"""
        if self.model is None:
            raise ValueError("يجب إنشاء النموذج أولاً")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.trainer:
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.verbose:
            logger.info(f"✅ Model loaded from {path}")
            if 'device_info' in checkpoint:
                logger.info(f"   Trained on: {checkpoint['device_info']['name']}")

    def load_state_dict(self, state_dict: Dict):
        """تحميل state_dict مباشرة (للتوافق مع الكود القديم)"""
        if self.model is None:
            raise ValueError("يجب إنشاء النموذج أولاً")

        self.model.load_state_dict(state_dict)

        if self.verbose:
            logger.info("✅ Model state loaded")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # System Info - معلومات النظام
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_system_info(self) -> Dict[str, Any]:
        """معلومات النظام الكاملة"""
        info = {
            "device": {
                "type": self.device.type,
                "name": self.device_manager.device_info.name,
                "memory_total_mb": self.device_manager.device_info.memory_total,
                "memory_available_mb": self.device_manager.device_info.memory_available,
            },
            "model": {
                "loaded": self.model is not None,
                "trainable": self.trainer is not None,
            },
            "advanced_features": ADVANCED_FEATURES,
        }

        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            info["model"]["parameters"] = total_params
            info["model"]["size_mb"] = total_params * 4 / 1024 / 1024

            # Add MegaBrain stats if available
            if hasattr(self.model, 'get_stats'):
                info["model"]["stats"] = self.model.get_stats()

        if ADVANCED_FEATURES and self.knowledge_graph:
            info["knowledge"] = self.get_knowledge_stats()

        return info

    def print_system_info(self):
        """طباعة معلومات النظام"""
        info = self.get_system_info()

        logger.info("\n" + "="*60)
        logger.info("🧠 Unified Noogh Brain - System Info")
        logger.info("="*60)
        logger.info(f"Device: {info['device']['type'].upper()} - {info['device']['name']}")
        logger.info(f"Memory: {info['device']['memory_available_mb']} MB / {info['device']['memory_total_mb']} MB")
        logger.info(f"Model Loaded: {'✅' if info['model']['loaded'] else '❌'}")

        if info['model']['loaded']:
            logger.info(f"  Parameters: {info['model'].get('parameters', 0):,}")
            logger.info(f"  Size: {info['model'].get('size_mb', 0):.2f} MB")

            if 'stats' in info['model']:
                stats = info['model']['stats']
                logger.info(f"  Model Type: {stats.get('model_name', 'Unknown')}")

        logger.info(f"Advanced Features: {'✅ Enabled' if info['advanced_features'] else '❌ Disabled'}")

        if info.get('knowledge'):
            logger.info(f"\nKnowledge Graph:")
            logger.info(f"  Nodes: {info['knowledge']['nodes']}")
            logger.info(f"  Edges: {info['knowledge']['edges']}")

        logger.info("="*60 + "\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hugging Face Integration - تكامل Hugging Face
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def hf_download_model(self, repo_id: str, revision: str = None) -> str:
        """
        📥 تنزيل نموذج من Hugging Face Hub
        Download model from Hugging Face Hub

        Args:
            repo_id: Repository ID (e.g., "meta-llama/Llama-3.1-8B")
            revision: Git revision (branch/tag/commit)

        Returns:
            Path to downloaded model

        Example:
            path = brain.hf_download_model("google/gemma-2-2b-it")
        """
        if not self.hf:
            raise RuntimeError("🤗 HF Hub not available - check installation")

        return self.hf.download_model(repo_id, revision)

    def hf_load_text_model(self, repo_id: str, use_4bit: bool = True, torch_dtype: str = "bfloat16"):
        """
        🤖 تحميل نموذج نصي من HF مع quantization
        Load text model from HF with quantization

        Args:
            repo_id: Repository ID
            use_4bit: Use 4-bit quantization (saves VRAM)
            torch_dtype: PyTorch dtype ("bfloat16", "float16")

        Returns:
            (tokenizer, model) tuple

        Example:
            tok, model = brain.hf_load_text_model(
                "google/gemma-2-2b-it",
                use_4bit=True
            )
        """
        if not self.hf:
            raise RuntimeError("🤗 HF Hub not available - check installation")

        quant = self.hf.make_quant_4bit() if use_4bit else None
        tok, model = self.hf.load_text_model(
            repo_id,
            device="auto",
            quantization=quant,
            torch_dtype=torch_dtype
        )

        if self.verbose:
            logger.info(f"✅ Loaded {repo_id}")
            if use_4bit:
                logger.info(f"   Quantization: 4-bit (NF4)")

        return tok, model

    def hf_push_checkpoint(self, local_dir: str, repo_id: str, private: bool = True):
        """
        📤 رفع checkpoint إلى HF Hub
        Push checkpoint to HF Hub

        Args:
            local_dir: Local checkpoint directory
            repo_id: Target repository ID
            private: Make repository private

        Returns:
            Full repository ID

        Example:
            repo = brain.hf_push_checkpoint(
                local_dir="/path/to/checkpoint",
                repo_id="my-model",
                private=True
            )
        """
        if not self.hf:
            raise RuntimeError("🤗 HF Hub not available - check installation")

        return self.hf.push_model_folder(
            local_dir,
            repo_id,
            private=private,
            commit_msg="Noogh auto-checkpoint"
        )

    def hf_list_models(self, query: Optional[str] = None, limit: int = 20):
        """
        📋 قائمة النماذج من HF Hub
        List models from HF Hub

        Args:
            query: Search query (e.g., "arabic", "llama")
            limit: Maximum number of results

        Returns:
            List of model info dicts

        Example:
            models = brain.hf_list_models(query="arabic", limit=10)
            for m in models:
                print(f"{m['id']}: {m['downloads']} downloads")
        """
        if not self.hf:
            raise RuntimeError("🤗 HF Hub not available - check installation")

        return self.hf.list_models(query=query, limit=limit)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HF Inference API - تكامل Inference API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def hf_generate(self, model: str, inputs: str, max_new_tokens: int = 50, **kwargs):
        """
        🌐 توليد نص عبر Cloud Inference API
        Generate text via HF Inference API

        Args:
            model: Model ID (e.g., "google/gemma-2-2b-it")
            inputs: Input prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text dict

        Example:
            result = brain.hf_generate(
                model="google/gemma-2-2b-it",
                inputs="What is the capital of France?",
                max_new_tokens=50
            )
            print(result['generated_text'])
        """
        if not self.hf_inference:
            raise RuntimeError("🌐 HF Inference API not available")

        return self.hf_inference.generate(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

    def hf_classify(self, model: str, inputs: str, top_k: Optional[int] = None):
        """
        🏷️  تصنيف نص عبر Cloud
        Classify text via HF Inference API

        Args:
            model: Classification model ID
            inputs: Input text
            top_k: Return top K predictions

        Returns:
            Classification results

        Example:
            result = brain.hf_classify(
                model="distilbert-base-uncased-finetuned-sst-2-english",
                inputs="I love this product!"
            )
        """
        if not self.hf_inference:
            raise RuntimeError("🌐 HF Inference API not available")

        return self.hf_inference.classify(model=model, inputs=inputs, top_k=top_k)

    def hf_embed(self, model: str, inputs: Union[str, List[str]]):
        """
        📊 Get embeddings via Cloud
        استخراج embeddings عبر Cloud

        Args:
            model: Embedding model ID
            inputs: Text or list of texts

        Returns:
            Embeddings

        Example:
            embeddings = brain.hf_embed(
                model="sentence-transformers/all-MiniLM-L6-v2",
                inputs="Hello world"
            )
        """
        if not self.hf_inference:
            raise RuntimeError("🌐 HF Inference API not available")

        return self.hf_inference.embed(model=model, inputs=inputs)

    def hf_inference_stats(self):
        """
        📊 إحصائيات استخدام Inference API
        Get Inference API usage statistics

        Returns:
            Usage stats dict

        Example:
            stats = brain.hf_inference_stats()
            print(f"Total calls: {stats['total_calls']}")
            print(f"Success rate: {stats['success_rate']:.2%}")
        """
        if not self.hf_inference:
            raise RuntimeError("🌐 HF Inference API not available")

        return self.hf_inference.get_stats()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Backward Compatibility - التوافق مع الكود القديم
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NooghBrain = UnifiedNooghBrain
EnhancedBrain = UnifiedNooghBrain
IntegratedNooghBrain = UnifiedNooghBrain


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions - دوال مساعدة
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_brain(device: str = "auto", verbose: bool = True, use_autonomous: bool = True) -> UnifiedNooghBrain:
    """
    إنشاء عقل Noogh موحد

    Args:
        device: الجهاز ("auto", "cuda", "cpu")
        verbose: طباعة معلومات مفصلة
        use_autonomous: استخدام النظام الذاتي (True = مفعّل)

    Usage:
        # Simple model
        brain = create_brain()
        brain.create_model('mlp', input_size=784, hidden_sizes=[256, 128], output_size=10)

        # MegaBrain with autonomous system
        brain = create_brain(use_autonomous=True)
        brain.create_mega_brain(config='small')
        brain.check_resources()  # Check VRAM before training
        brain.train(train_data, train_labels, test_data, test_labels, epochs=10)
    """
    return UnifiedNooghBrain(device=device, verbose=verbose, use_autonomous=use_autonomous)


def create_mega_brain(
    config: str = "small",
    device: str = "auto",
    verbose: bool = True,
    use_autonomous: bool = True
) -> UnifiedNooghBrain:
    """
    إنشاء MegaBrain V5 مباشرة

    Configs: micro, tiny, small, base, medium, large, ultra, insane

    Args:
        config: حجم النموذج
        device: الجهاز ("auto", "cuda", "cpu")
        verbose: طباعة معلومات مفصلة
        use_autonomous: استخدام النظام الذاتي (True = تدريب ذكي)

    Usage:
        # Smart training with autonomous system
        brain = create_mega_brain(config='small', use_autonomous=True)

        # Check if ready
        if brain.is_ready_for_training(estimated_vram=4.0):
            brain.train(train_data, train_labels, test_data, test_labels)
    """
    brain = UnifiedNooghBrain(device=device, verbose=verbose, use_autonomous=use_autonomous)
    brain.create_mega_brain(config=config)
    return brain


if __name__ == "__main__":
    # اختبار سريع
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("🧪 Testing Unified Noogh Brain...\n")

    # Test 1: Basic brain
    brain = create_brain()
    brain.print_system_info()

    # Test 2: MegaBrain (micro config for quick test)
    logger.info("\n" + "="*60)
    logger.info("Testing MegaBrain V5 (micro config)")
    logger.info("="*60 + "\n")

    mega_brain = create_mega_brain(config='micro')

    # Test forward pass
    test_input = torch.randn(2, 1024).to(mega_brain.device)
    output = mega_brain.model(test_input)

    logger.info(f"Test input shape: {test_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"\n✅ Unified Noogh Brain ready!")
