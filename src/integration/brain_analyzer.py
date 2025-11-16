"""
Brain Structure Analyzer - Advanced Neural Network Inspection
تحليل معمق لهيكل الدماغ العصبي MegaBrain V5
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BrainAnalyzer:
    """محلل متقدم لهيكل وأداء الدماغ العصبي"""

    def __init__(self, checkpoint_dir: str = "/home/noogh/brain_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.brain_model = None
        self.training_history = None

    def load_brain(self, model_path: Optional[str] = None) -> bool:
        """تحميل نموذج الدماغ للتحليل"""
        try:
            if model_path is None:
                model_path = self.checkpoint_dir / "best_model.pt"
            else:
                model_path = Path(model_path)

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return False

            self.brain_model = torch.load(model_path, map_location='cpu')
            logger.info(f"✅ Brain model loaded: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load brain: {e}")
            return False

    def load_training_history(self) -> bool:
        """تحميل سجل التدريب"""
        try:
            history_path = self.checkpoint_dir / "training_history.json"
            if not history_path.exists():
                return False

            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            return True

        except Exception as e:
            logger.error(f"Failed to load training history: {e}")
            return False

    def analyze_architecture(self) -> Dict[str, Any]:
        """تحليل معماري كامل للشبكة العصبية"""
        if self.brain_model is None:
            self.load_brain()

        if self.brain_model is None:
            return {"error": "No brain model loaded"}

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "MegaBrain V5",
            "layers": [],
            "total_parameters": 0,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
            "model_size_mb": 0,
            "architecture_summary": {}
        }

        try:
            # تحليل الطبقات
            layer_types = {}
            total_params = 0
            trainable_params = 0

            for name, param in self.brain_model.items():
                if isinstance(param, torch.Tensor):
                    params = param.numel()
                    total_params += params

                    # تحديد نوع الطبقة
                    layer_type = self._identify_layer_type(name)
                    if layer_type not in layer_types:
                        layer_types[layer_type] = {
                            "count": 0,
                            "parameters": 0,
                            "layers": []
                        }

                    layer_types[layer_type]["count"] += 1
                    layer_types[layer_type]["parameters"] += params
                    layer_types[layer_type]["layers"].append({
                        "name": name,
                        "shape": list(param.shape),
                        "parameters": params,
                        "dtype": str(param.dtype)
                    })

                    analysis["layers"].append({
                        "name": name,
                        "type": layer_type,
                        "shape": list(param.shape),
                        "parameters": params,
                        "dtype": str(param.dtype),
                        "device": str(param.device)
                    })

            analysis["total_parameters"] = total_params
            analysis["trainable_parameters"] = total_params  # All params trainable by default
            analysis["architecture_summary"] = layer_types

            # حساب حجم النموذج
            model_path = self.checkpoint_dir / "best_model.pt"
            if model_path.exists():
                analysis["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)

            return analysis

        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}")
            return {"error": str(e)}

    def _identify_layer_type(self, layer_name: str) -> str:
        """تحديد نوع الطبقة من اسمها"""
        name_lower = layer_name.lower()

        if 'embedding' in name_lower:
            return "Embedding"
        elif 'attention' in name_lower:
            return "Attention"
        elif 'transformer' in name_lower:
            return "Transformer"
        elif 'conv' in name_lower:
            return "Convolution"
        elif 'bn' in name_lower or 'batch_norm' in name_lower:
            return "BatchNorm"
        elif 'linear' in name_lower or 'fc' in name_lower:
            return "Linear"
        elif 'lstm' in name_lower or 'gru' in name_lower:
            return "RNN"
        elif 'weight' in name_lower:
            return "Weight"
        elif 'bias' in name_lower:
            return "Bias"
        else:
            return "Other"

    def analyze_training_progress(self) -> Dict[str, Any]:
        """تحليل تقدم التدريب والأداء"""
        if self.training_history is None:
            self.load_training_history()

        if self.training_history is None:
            return {"error": "No training history available"}

        try:
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "training_completed": True,
                "total_epochs": len(self.training_history.get("train_losses", [])),
                "metrics": {},
                "improvements": {},
                "performance_trend": ""
            }

            train_losses = self.training_history.get("train_loss", self.training_history.get("train_losses", []))
            val_losses = self.training_history.get("val_loss", self.training_history.get("val_losses", []))

            if train_losses and val_losses:
                # حساب التحسينات
                initial_train_loss = train_losses[0] if train_losses else 0
                final_train_loss = train_losses[-1] if train_losses else 0
                initial_val_loss = val_losses[0] if val_losses else 0
                final_val_loss = val_losses[-1] if val_losses else 0

                train_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss * 100) if initial_train_loss > 0 else 0
                val_improvement = ((initial_val_loss - final_val_loss) / initial_val_loss * 100) if initial_val_loss > 0 else 0

                analysis["metrics"] = {
                    "initial_train_loss": round(initial_train_loss, 4),
                    "final_train_loss": round(final_train_loss, 4),
                    "initial_val_loss": round(initial_val_loss, 4),
                    "final_val_loss": round(final_val_loss, 4),
                    "best_val_loss": round(min(val_losses), 4),
                    "avg_train_loss": round(sum(train_losses) / len(train_losses), 4),
                    "avg_val_loss": round(sum(val_losses) / len(val_losses), 4)
                }

                analysis["improvements"] = {
                    "train_loss_improvement_pct": round(train_improvement, 2),
                    "val_loss_improvement_pct": round(val_improvement, 2),
                    "generalization_gap": round(final_val_loss - final_train_loss, 4)
                }

                # تحديد اتجاه الأداء
                if val_improvement > 80:
                    analysis["performance_trend"] = "ممتاز - تحسن كبير جداً"
                elif val_improvement > 60:
                    analysis["performance_trend"] = "جيد جداً - تحسن ملحوظ"
                elif val_improvement > 40:
                    analysis["performance_trend"] = "جيد - تحسن مستمر"
                elif val_improvement > 20:
                    analysis["performance_trend"] = "مقبول - تحسن معتدل"
                else:
                    analysis["performance_trend"] = "ضعيف - تحسن محدود"

            return analysis

        except Exception as e:
            logger.error(f"Training analysis failed: {e}")
            return {"error": str(e)}

    def get_brain_health_score(self) -> Dict[str, Any]:
        """حساب مؤشر صحة الدماغ الشامل"""
        architecture = self.analyze_architecture()
        training = self.analyze_training_progress()

        health_score = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": 0,
            "components": {},
            "status": "unknown",
            "recommendations": []
        }

        try:
            scores = {}

            # 1. Architecture Health (30%)
            if "error" not in architecture:
                arch_score = 100  # Base score
                total_params = architecture.get("total_parameters", 0)

                # Penalize if too small or too large
                if total_params < 1_000_000:
                    arch_score -= 20
                    health_score["recommendations"].append("النموذج صغير جداً - فكر في زيادة السعة")
                elif total_params > 100_000_000:
                    arch_score -= 10
                    health_score["recommendations"].append("النموذج كبير جداً - قد يحتاج optimization")

                scores["architecture"] = max(0, arch_score)

            # 2. Training Performance (40%)
            if "error" not in training and "improvements" in training:
                val_improvement = training["improvements"].get("val_loss_improvement_pct", 0)

                if val_improvement > 80:
                    scores["training"] = 100
                elif val_improvement > 60:
                    scores["training"] = 85
                elif val_improvement > 40:
                    scores["training"] = 70
                elif val_improvement > 20:
                    scores["training"] = 50
                else:
                    scores["training"] = 30
                    health_score["recommendations"].append("التحسن في التدريب محدود - فكر في تعديل hyperparameters")

            # 3. Generalization (30%)
            if "error" not in training and "improvements" in training:
                gap = abs(training["improvements"].get("generalization_gap", 0))

                if gap < 0.02:
                    scores["generalization"] = 100
                elif gap < 0.05:
                    scores["generalization"] = 85
                elif gap < 0.1:
                    scores["generalization"] = 70
                elif gap < 0.2:
                    scores["generalization"] = 50
                else:
                    scores["generalization"] = 30
                    health_score["recommendations"].append("فجوة التعميم كبيرة - قد يكون هناك overfitting")

            # حساب الدرجة الإجمالية
            if scores:
                weights = {
                    "architecture": 0.3,
                    "training": 0.4,
                    "generalization": 0.3
                }

                overall = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights.keys())
                health_score["overall_health"] = round(overall, 1)
                health_score["components"] = scores

                # تحديد الحالة
                if overall >= 90:
                    health_score["status"] = "ممتاز 🟢"
                elif overall >= 75:
                    health_score["status"] = "جيد جداً 🟢"
                elif overall >= 60:
                    health_score["status"] = "جيد 🟡"
                elif overall >= 40:
                    health_score["status"] = "مقبول 🟡"
                else:
                    health_score["status"] = "يحتاج تحسين 🔴"

            return health_score

        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return {"error": str(e)}

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """تقرير شامل عن حالة الدماغ"""
        return {
            "timestamp": datetime.now().isoformat(),
            "architecture": self.analyze_architecture(),
            "training_progress": self.analyze_training_progress(),
            "health_score": self.get_brain_health_score()
        }


# مثال على الاستخدام
if __name__ == "__main__":
    analyzer = BrainAnalyzer()
    report = analyzer.generate_comprehensive_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
