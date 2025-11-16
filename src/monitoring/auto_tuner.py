#!/usr/bin/env python3
"""
🎯 Auto-Tuning System
نظام الضبط الذاتي

Automatically optimizes training parameters based on:
- ML predictions
- Evaluation feedback
- Historical performance
- System constraints

Features:
✅ Dynamic VRAM allocation
✅ Batch size optimization
✅ Safety margin adjustment
✅ Learning rate suggestions
✅ Device selection optimization
✅ Ministers pause/unpause decisions
✅ Feedback loop integration
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parameter Adjustment Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdjustmentType(str, Enum):
    """Types of parameter adjustments"""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"


class ParameterType(str, Enum):
    """Types of tunable parameters"""
    VRAM_ALLOCATION = "vram_allocation"
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"
    SAFETY_MARGIN = "safety_margin"
    DEVICE = "device"
    MINISTERS_PAUSE = "ministers_pause"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tuning Recommendation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TuningRecommendation:
    """A single parameter tuning recommendation"""
    parameter: ParameterType
    adjustment: AdjustmentType
    current_value: Any
    recommended_value: Any
    confidence: float  # 0.0 to 1.0
    reason: str
    impact: str  # "low", "medium", "high"
    timestamp: str


@dataclass
class TuningProfile:
    """Complete tuning profile for a training session"""
    model_name: str
    vram_gb: float
    batch_size: Optional[int]
    learning_rate: Optional[float]
    safety_margin: float
    device: str
    pause_ministers: bool
    ministers_count: int
    recommendations: List[TuningRecommendation]
    confidence_overall: float
    estimated_success_rate: float  # 0.0 to 1.0
    timestamp: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-Tuning System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AutoTuningSystem:
    """
    🎯 Autonomous Parameter Tuning System

    Automatically optimizes training parameters based on:
    - ML predictions (VRAM, duration)
    - Evaluation feedback (success/failure patterns)
    - Historical performance (similar models)
    - System constraints (available resources)
    """

    def __init__(
        self,
        storage_path: str = "/home/noogh/projects/noogh_unified_system/logs",
        verbose: bool = True
    ):
        """
        Initialize auto-tuning system

        Args:
            storage_path: Where to store tuning profiles
            verbose: Enable detailed logging
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.tuning_file = self.storage_path / "tuning_profiles.jsonl"
        self.verbose = verbose

        # Default constraints
        self.constraints = {
            "max_vram_gb": 24.0,  # RTX 3090 limit
            "min_batch_size": 1,
            "max_batch_size": 128,
            "min_learning_rate": 1e-6,
            "max_learning_rate": 1e-2,
            "min_safety_margin": 1.05,
            "max_safety_margin": 1.5,
        }

        if self.verbose:
            logger.info(f"🎯 Auto-Tuning System initialized")
            logger.info(f"   Storage: {self.tuning_file}")


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Main Tuning Interface
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def tune_for_model(
        self,
        model_name: str,
        epochs: int = 10,
        ml_predictions: Optional[Dict[str, Any]] = None,
        evaluation_stats: Optional[Dict[str, Any]] = None,
        current_resources: Optional[Dict[str, Any]] = None,
        user_overrides: Optional[Dict[str, Any]] = None
    ) -> TuningProfile:
        """
        Generate optimized parameters for training a model

        Args:
            model_name: Name of model to train
            epochs: Number of training epochs
            ml_predictions: ML predictor results (VRAM, duration)
            evaluation_stats: Recent evaluation statistics
            current_resources: Current system resource status
            user_overrides: User-specified parameter overrides

        Returns:
            TuningProfile with optimized parameters and recommendations
        """
        if self.verbose:
            logger.info(f"🎯 Tuning parameters for: {model_name}")

        recommendations = []

        # Start with defaults or user overrides
        params = {
            "vram_gb": user_overrides.get("vram_gb") if user_overrides else None,
            "batch_size": user_overrides.get("batch_size") if user_overrides else 32,
            "learning_rate": user_overrides.get("learning_rate") if user_overrides else 1e-4,
            "safety_margin": user_overrides.get("safety_margin") if user_overrides else 1.1,
            "device": user_overrides.get("device") if user_overrides else "gpu",
            "pause_ministers": user_overrides.get("pause_ministers") if user_overrides else True,
            "ministers_count": 0
        }

        # 1. VRAM Allocation
        vram_rec = self._tune_vram_allocation(
            model_name, epochs, ml_predictions, evaluation_stats,
            current_resources, params["vram_gb"]
        )
        if vram_rec:
            recommendations.append(vram_rec)
            params["vram_gb"] = vram_rec.recommended_value
        elif params["vram_gb"] is None:
            # Fallback to default estimation
            params["vram_gb"] = 4.0

        # 2. Safety Margin
        margin_rec = self._tune_safety_margin(
            model_name, evaluation_stats, params["safety_margin"]
        )
        if margin_rec:
            recommendations.append(margin_rec)
            params["safety_margin"] = margin_rec.recommended_value

        # 3. Batch Size
        batch_rec = self._tune_batch_size(
            model_name, params["vram_gb"], evaluation_stats, params["batch_size"]
        )
        if batch_rec:
            recommendations.append(batch_rec)
            params["batch_size"] = batch_rec.recommended_value

        # 4. Learning Rate
        lr_rec = self._tune_learning_rate(
            model_name, evaluation_stats, params["learning_rate"]
        )
        if lr_rec:
            recommendations.append(lr_rec)
            params["learning_rate"] = lr_rec.recommended_value

        # 5. Device Selection
        device_rec = self._tune_device_selection(
            params["vram_gb"], current_resources, params["device"]
        )
        if device_rec:
            recommendations.append(device_rec)
            params["device"] = device_rec.recommended_value

        # 6. Ministers Pause Decision
        ministers_rec = self._tune_ministers_pause(
            params["vram_gb"], current_resources, params["pause_ministers"]
        )
        if ministers_rec:
            recommendations.append(ministers_rec)
            params["pause_ministers"] = ministers_rec.recommended_value
            params["ministers_count"] = current_resources.get("ministers_count", 0) if params["pause_ministers"] else 0

        # Calculate overall confidence and success rate
        confidence_overall = self._calculate_overall_confidence(recommendations)
        success_rate = self._estimate_success_rate(
            evaluation_stats, params["vram_gb"], params["safety_margin"]
        )

        # Create tuning profile
        profile = TuningProfile(
            model_name=model_name,
            vram_gb=params["vram_gb"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            safety_margin=params["safety_margin"],
            device=params["device"],
            pause_ministers=params["pause_ministers"],
            ministers_count=params["ministers_count"],
            recommendations=recommendations,
            confidence_overall=confidence_overall,
            estimated_success_rate=success_rate,
            timestamp=datetime.now().isoformat()
        )

        # Store profile
        self._save_profile(profile)

        if self.verbose:
            logger.info(f"✅ Tuning complete: {len(recommendations)} recommendations")
            logger.info(f"   VRAM: {params['vram_gb']:.2f} GB")
            logger.info(f"   Success rate: {success_rate:.1%}")

        return profile


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Individual Parameter Tuners
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _tune_vram_allocation(
        self,
        model_name: str,
        epochs: int,
        ml_predictions: Optional[Dict],
        evaluation_stats: Optional[Dict],
        current_resources: Optional[Dict],
        current_value: Optional[float]
    ) -> Optional[TuningRecommendation]:
        """Tune VRAM allocation using ML predictions and historical data"""

        # If ML prediction available, use it
        if ml_predictions and "with_margin_gb" in ml_predictions:
            ml_vram = ml_predictions["with_margin_gb"]
            confidence = ml_predictions.get("confidence", 0.8)

            if current_value is None or abs(ml_vram - current_value) > 0.5:
                return TuningRecommendation(
                    parameter=ParameterType.VRAM_ALLOCATION,
                    adjustment=AdjustmentType.INCREASE if ml_vram > (current_value or 0) else AdjustmentType.DECREASE,
                    current_value=current_value,
                    recommended_value=ml_vram,
                    confidence=confidence,
                    reason=f"ML predictor estimates {ml_vram:.2f} GB based on historical data",
                    impact="high",
                    timestamp=datetime.now().isoformat()
                )

        # Fallback to evaluation statistics
        if evaluation_stats and "avg_vram_peak_gb" in evaluation_stats:
            avg_vram = evaluation_stats["avg_vram_peak_gb"]

            if current_value is None or abs(avg_vram - current_value) > 0.5:
                return TuningRecommendation(
                    parameter=ParameterType.VRAM_ALLOCATION,
                    adjustment=AdjustmentType.INCREASE if avg_vram > (current_value or 0) else AdjustmentType.DECREASE,
                    current_value=current_value,
                    recommended_value=avg_vram * 1.1,  # +10% safety
                    confidence=0.6,
                    reason=f"Based on historical average: {avg_vram:.2f} GB",
                    impact="high",
                    timestamp=datetime.now().isoformat()
                )

        return None

    def _tune_safety_margin(
        self,
        model_name: str,
        evaluation_stats: Optional[Dict],
        current_value: float
    ) -> Optional[TuningRecommendation]:
        """Adjust safety margin based on VRAM estimation accuracy"""

        if not evaluation_stats:
            return None

        # Check VRAM estimation error
        avg_error_percent = evaluation_stats.get("avg_vram_error_percent", 0)

        # If error is high (>15%), increase safety margin
        if avg_error_percent > 15 and current_value < 1.3:
            new_margin = min(current_value + 0.1, self.constraints["max_safety_margin"])
            return TuningRecommendation(
                parameter=ParameterType.SAFETY_MARGIN,
                adjustment=AdjustmentType.INCREASE,
                current_value=current_value,
                recommended_value=new_margin,
                confidence=0.85,
                reason=f"VRAM estimation error is {avg_error_percent:.1f}% - increasing safety margin",
                impact="medium",
                timestamp=datetime.now().isoformat()
            )

        # If error is low (<5%) and margin is high, decrease
        elif avg_error_percent < 5 and current_value > 1.1:
            new_margin = max(current_value - 0.05, self.constraints["min_safety_margin"])
            return TuningRecommendation(
                parameter=ParameterType.SAFETY_MARGIN,
                adjustment=AdjustmentType.DECREASE,
                current_value=current_value,
                recommended_value=new_margin,
                confidence=0.75,
                reason=f"VRAM estimation is accurate ({avg_error_percent:.1f}%) - can reduce margin",
                impact="low",
                timestamp=datetime.now().isoformat()
            )

        return None

    def _tune_batch_size(
        self,
        model_name: str,
        vram_gb: float,
        evaluation_stats: Optional[Dict],
        current_value: int
    ) -> Optional[TuningRecommendation]:
        """Optimize batch size based on VRAM and historical performance"""

        # Simple heuristic: larger VRAM → larger batch size
        if vram_gb >= 16:
            recommended = 64
        elif vram_gb >= 8:
            recommended = 32
        elif vram_gb >= 4:
            recommended = 16
        else:
            recommended = 8

        # Clamp to constraints
        recommended = max(
            self.constraints["min_batch_size"],
            min(recommended, self.constraints["max_batch_size"])
        )

        if recommended != current_value:
            return TuningRecommendation(
                parameter=ParameterType.BATCH_SIZE,
                adjustment=AdjustmentType.INCREASE if recommended > current_value else AdjustmentType.DECREASE,
                current_value=current_value,
                recommended_value=recommended,
                confidence=0.7,
                reason=f"Optimal batch size for {vram_gb:.1f} GB VRAM",
                impact="medium",
                timestamp=datetime.now().isoformat()
            )

        return None

    def _tune_learning_rate(
        self,
        model_name: str,
        evaluation_stats: Optional[Dict],
        current_value: float
    ) -> Optional[TuningRecommendation]:
        """Suggest learning rate based on model type and historical data"""

        # Model-specific learning rates (simple heuristics)
        model_lower = model_name.lower()

        if "gpt" in model_lower or "llama" in model_lower:
            recommended = 5e-5  # Lower for large language models
        elif "bert" in model_lower or "roberta" in model_lower:
            recommended = 2e-5  # Even lower for BERT-style
        elif "resnet" in model_lower or "vit" in model_lower:
            recommended = 1e-4  # Moderate for vision models
        else:
            recommended = 1e-4  # Default

        # Clamp to constraints
        recommended = max(
            self.constraints["min_learning_rate"],
            min(recommended, self.constraints["max_learning_rate"])
        )

        if abs(recommended - current_value) > current_value * 0.2:  # >20% difference
            return TuningRecommendation(
                parameter=ParameterType.LEARNING_RATE,
                adjustment=AdjustmentType.INCREASE if recommended > current_value else AdjustmentType.DECREASE,
                current_value=current_value,
                recommended_value=recommended,
                confidence=0.65,
                reason=f"Recommended learning rate for {model_name} type models",
                impact="medium",
                timestamp=datetime.now().isoformat()
            )

        return None

    def _tune_device_selection(
        self,
        vram_gb: float,
        current_resources: Optional[Dict],
        current_value: str
    ) -> Optional[TuningRecommendation]:
        """Choose optimal device (GPU vs CPU) based on requirements"""

        if not current_resources:
            return None

        available_vram_percent = current_resources.get("vram_available_percent", 100)
        available_vram_gb = current_resources.get("vram_available_gb", 24)

        # If GPU doesn't have enough VRAM, suggest CPU
        if vram_gb > available_vram_gb and current_value == "gpu":
            return TuningRecommendation(
                parameter=ParameterType.DEVICE,
                adjustment=AdjustmentType.DECREASE,
                current_value=current_value,
                recommended_value="cpu",
                confidence=0.9,
                reason=f"GPU VRAM insufficient ({available_vram_gb:.1f} GB available, {vram_gb:.1f} GB needed)",
                impact="high",
                timestamp=datetime.now().isoformat()
            )

        # If using CPU but GPU has plenty of VRAM, suggest GPU
        elif current_value == "cpu" and available_vram_gb > vram_gb * 1.5:
            return TuningRecommendation(
                parameter=ParameterType.DEVICE,
                adjustment=AdjustmentType.INCREASE,
                current_value=current_value,
                recommended_value="gpu",
                confidence=0.85,
                reason=f"GPU has sufficient VRAM ({available_vram_gb:.1f} GB available)",
                impact="high",
                timestamp=datetime.now().isoformat()
            )

        return None

    def _tune_ministers_pause(
        self,
        vram_gb: float,
        current_resources: Optional[Dict],
        current_value: bool
    ) -> Optional[TuningRecommendation]:
        """Decide whether to pause ministers based on VRAM needs"""

        if not current_resources:
            return None

        available_vram_percent = current_resources.get("vram_available_percent", 100)

        # If VRAM is tight (<30% available) and ministers not paused, suggest pausing
        if available_vram_percent < 30 and not current_value:
            return TuningRecommendation(
                parameter=ParameterType.MINISTERS_PAUSE,
                adjustment=AdjustmentType.INCREASE,
                current_value=current_value,
                recommended_value=True,
                confidence=0.9,
                reason=f"Low VRAM availability ({available_vram_percent:.1f}%) - pause ministers to free resources",
                impact="high",
                timestamp=datetime.now().isoformat()
            )

        # If VRAM is abundant (>70% available) and ministers paused, suggest unpausing
        elif available_vram_percent > 70 and current_value and vram_gb < 8:
            return TuningRecommendation(
                parameter=ParameterType.MINISTERS_PAUSE,
                adjustment=AdjustmentType.DECREASE,
                current_value=current_value,
                recommended_value=False,
                confidence=0.7,
                reason=f"High VRAM availability ({available_vram_percent:.1f}%) - can keep ministers running",
                impact="low",
                timestamp=datetime.now().isoformat()
            )

        return None


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Confidence & Success Rate Estimation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calculate_overall_confidence(
        self,
        recommendations: List[TuningRecommendation]
    ) -> float:
        """Calculate overall confidence from all recommendations"""
        if not recommendations:
            return 0.5  # Neutral confidence

        # Weighted average based on impact
        impact_weights = {"low": 0.5, "medium": 1.0, "high": 1.5}

        total_weighted_conf = 0.0
        total_weight = 0.0

        for rec in recommendations:
            weight = impact_weights.get(rec.impact, 1.0)
            total_weighted_conf += rec.confidence * weight
            total_weight += weight

        return total_weighted_conf / total_weight if total_weight > 0 else 0.5

    def _estimate_success_rate(
        self,
        evaluation_stats: Optional[Dict],
        vram_gb: float,
        safety_margin: float
    ) -> float:
        """Estimate probability of successful training"""

        # Base success rate from historical data
        if evaluation_stats and "success_rate_percent" in evaluation_stats:
            base_rate = evaluation_stats["success_rate_percent"] / 100.0
        else:
            base_rate = 0.7  # Default 70%

        # Adjust based on safety margin
        margin_bonus = min((safety_margin - 1.0) * 0.2, 0.2)  # Up to +20%

        # Adjust based on VRAM allocation
        if vram_gb >= 16:
            vram_bonus = 0.1  # +10% for high VRAM
        elif vram_gb < 4:
            vram_bonus = -0.1  # -10% for low VRAM
        else:
            vram_bonus = 0.0

        estimated_rate = min(base_rate + margin_bonus + vram_bonus, 1.0)
        return max(estimated_rate, 0.0)


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Storage
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_profile(self, profile: TuningProfile):
        """Save tuning profile to JSONL storage"""
        try:
            # Convert to dict
            profile_dict = asdict(profile)

            # Append to JSONL
            with open(self.tuning_file, 'a') as f:
                f.write(json.dumps(profile_dict) + '\n')

            if self.verbose:
                logger.debug(f"💾 Tuning profile saved: {profile.model_name}")

        except Exception as e:
            logger.error(f"Failed to save tuning profile: {e}")

    def get_profile_history(
        self,
        model_name: Optional[str] = None,
        days: int = 30
    ) -> List[TuningProfile]:
        """Get historical tuning profiles"""
        if not self.tuning_file.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        profiles = []

        try:
            with open(self.tuning_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)

                        # Parse timestamp
                        timestamp = datetime.fromisoformat(data["timestamp"])

                        # Filter by date and model
                        if timestamp < cutoff_date:
                            continue

                        if model_name and data["model_name"] != model_name:
                            continue

                        # Reconstruct TuningProfile
                        recommendations = [
                            TuningRecommendation(**rec)
                            for rec in data["recommendations"]
                        ]

                        profile = TuningProfile(
                            model_name=data["model_name"],
                            vram_gb=data["vram_gb"],
                            batch_size=data["batch_size"],
                            learning_rate=data["learning_rate"],
                            safety_margin=data["safety_margin"],
                            device=data["device"],
                            pause_ministers=data["pause_ministers"],
                            ministers_count=data["ministers_count"],
                            recommendations=recommendations,
                            confidence_overall=data["confidence_overall"],
                            estimated_success_rate=data["estimated_success_rate"],
                            timestamp=data["timestamp"]
                        )

                        profiles.append(profile)

        except Exception as e:
            logger.error(f"Failed to load tuning profiles: {e}")

        return profiles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Instance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_auto_tuner: Optional[AutoTuningSystem] = None

def get_auto_tuner(verbose: bool = True) -> AutoTuningSystem:
    """Get or create global auto-tuning system instance"""
    global _auto_tuner
    if _auto_tuner is None:
        _auto_tuner = AutoTuningSystem(verbose=verbose)
    return _auto_tuner
