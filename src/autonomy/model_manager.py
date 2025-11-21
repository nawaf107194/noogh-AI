"""
🔄 Model Manager - Auto-Versioning & Rollback System
Manages model versions, performance tracking, and automatic rollback

Features:
- Model versioning with timestamps
- Performance comparison between versions
- Automatic rollback on degradation
- Model metadata tracking
- Integration with monitoring system
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import shutil
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str  # e.g., "v2025-11-10_06-30-00"
    timestamp: str
    model_path: str
    performance_metrics: Dict[str, float]
    training_params: Dict[str, Any]
    is_active: bool = False
    is_baseline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    accuracy: float  # 0-1
    avg_response_time_ms: float
    cognition_score: float  # 0-1
    success_rate: float  # 0-1
    memory_usage_mb: float
    test_samples: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def is_better_than(self, other: 'PerformanceMetrics',
                       threshold: float = 0.05) -> bool:
        """
        Check if this model is better than another

        Args:
            other: Other performance metrics to compare with
            threshold: Minimum improvement required (5% by default)

        Returns:
            True if this model is significantly better
        """
        # Weighted comparison
        score_diff = self.cognition_score - other.cognition_score
        accuracy_diff = self.accuracy - other.accuracy
        speed_ratio = other.avg_response_time_ms / max(self.avg_response_time_ms, 1)

        # Overall improvement score
        improvement = (
            score_diff * 0.5 +           # 50% weight on cognition
            accuracy_diff * 0.3 +         # 30% weight on accuracy
            (speed_ratio - 1) * 0.2       # 20% weight on speed
        )

        return improvement >= threshold


class ModelManager:
    """
    Model version manager with auto-rollback

    Manages:
    - Model versioning and storage
    - Performance tracking
    - Active model selection
    - Automatic rollback on degradation
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model manager

        Args:
            models_dir: Directory to store model versions
        """
        self.models_dir = models_dir or (PROJECT_ROOT / "models" / "versions")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.models_dir / "versions_metadata.json"
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version: Optional[ModelVersion] = None

        self._load_metadata()

    def _load_metadata(self):
        """Load versions metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load versions
                for version_data in data.get('versions', []):
                    version = ModelVersion(**version_data)
                    self.versions[version.version_id] = version

                    if version.is_active:
                        self.active_version = version

                logger.info(f"✅ Loaded {len(self.versions)} model versions")

            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        else:
            logger.info("📝 No existing metadata, starting fresh")

    def _save_metadata(self):
        """Save versions metadata to disk"""
        try:
            data = {
                'versions': [v.to_dict() for v in self.versions.values()],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("✅ Metadata saved")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def create_version_id(self) -> str:
        """Generate unique version ID based on timestamp"""
        return f"v{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    def register_model(
        self,
        model_path: str,
        performance_metrics: PerformanceMetrics,
        training_params: Optional[Dict[str, Any]] = None,
        set_active: bool = False
    ) -> ModelVersion:
        """
        Register a new model version

        Args:
            model_path: Path to the trained model
            performance_metrics: Model performance metrics
            training_params: Training parameters used
            set_active: Whether to set as active version

        Returns:
            ModelVersion object
        """
        version_id = self.create_version_id()

        # Copy model to versions directory
        version_dir = self.models_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            if model_path_obj.is_dir():
                # Copy directory
                dest_path = version_dir / model_path_obj.name
                shutil.copytree(model_path_obj, dest_path, dirs_exist_ok=True)
            else:
                # Copy file
                dest_path = version_dir / model_path_obj.name
                shutil.copy2(model_path_obj, dest_path)

            stored_model_path = str(dest_path)
        else:
            logger.warning(f"Model path not found: {model_path}, storing path only")
            stored_model_path = model_path

        # Create version
        version = ModelVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            model_path=stored_model_path,
            performance_metrics=performance_metrics.to_dict(),
            training_params=training_params or {},
            is_active=set_active,
            is_baseline=len(self.versions) == 0  # First model is baseline
        )

        # Store version
        self.versions[version_id] = version

        # Update active version
        if set_active:
            self._set_active_version(version_id)

        self._save_metadata()

        logger.info(f"✅ Registered model version: {version_id}")
        logger.info(f"   Cognition: {performance_metrics.cognition_score:.3f}")
        logger.info(f"   Accuracy: {performance_metrics.accuracy:.3f}")
        logger.info(f"   Response Time: {performance_metrics.avg_response_time_ms:.1f}ms")

        return version

    def _set_active_version(self, version_id: str):
        """Set a version as active"""
        # Deactivate all versions
        for v in self.versions.values():
            v.is_active = False

        # Activate specified version
        if version_id in self.versions:
            self.versions[version_id].is_active = True
            self.active_version = self.versions[version_id]
            logger.info(f"✅ Set active version: {version_id}")
        else:
            logger.error(f"Version not found: {version_id}")

    def compare_with_active(
        self,
        new_metrics: PerformanceMetrics
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Compare new model with active version

        Args:
            new_metrics: Performance metrics of new model

        Returns:
            (should_upgrade, reason, comparison_details)
        """
        if not self.active_version:
            return True, "No active version, upgrading", {}

        # Get active version metrics
        active_metrics_dict = self.active_version.performance_metrics
        active_metrics = PerformanceMetrics(**active_metrics_dict)

        # Compare
        is_better = new_metrics.is_better_than(active_metrics)

        # Calculate differences
        comparison = {
            'cognition_diff': new_metrics.cognition_score - active_metrics.cognition_score,
            'accuracy_diff': new_metrics.accuracy - active_metrics.accuracy,
            'speed_ratio': active_metrics.avg_response_time_ms / max(new_metrics.avg_response_time_ms, 1),
            'new_cognition': new_metrics.cognition_score,
            'old_cognition': active_metrics.cognition_score,
            'new_accuracy': new_metrics.accuracy,
            'old_accuracy': active_metrics.accuracy,
        }

        if is_better:
            reason = f"New model is better (cognition: {comparison['cognition_diff']:+.3f}, accuracy: {comparison['accuracy_diff']:+.3f})"
        else:
            reason = f"New model is not better (cognition: {comparison['cognition_diff']:+.3f}, accuracy: {comparison['accuracy_diff']:+.3f})"

        return is_better, reason, comparison

    def rollback_to_previous(self) -> Optional[ModelVersion]:
        """
        Rollback to previous version

        Returns:
            Previous version if successful, None otherwise
        """
        if not self.active_version:
            logger.error("No active version to rollback from")
            return None

        # Find previous version (by timestamp)
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.timestamp,
            reverse=True
        )

        # Find current active index
        current_idx = next(
            (i for i, v in enumerate(sorted_versions) if v.version_id == self.active_version.version_id),
            -1
        )

        if current_idx < 0 or current_idx >= len(sorted_versions) - 1:
            logger.error("No previous version available for rollback")
            return None

        # Get previous version
        previous_version = sorted_versions[current_idx + 1]

        # Set as active
        self._set_active_version(previous_version.version_id)
        self._save_metadata()

        logger.warning(f"⚠️ Rolled back to version: {previous_version.version_id}")

        return previous_version

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific version"""
        return self.versions.get(version_id)

    def get_all_versions(self) -> List[ModelVersion]:
        """Get all versions sorted by timestamp (newest first)"""
        return sorted(
            self.versions.values(),
            key=lambda v: v.timestamp,
            reverse=True
        )

    def get_baseline_version(self) -> Optional[ModelVersion]:
        """Get the baseline version (first model)"""
        return next(
            (v for v in self.versions.values() if v.is_baseline),
            None
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get model versioning statistics"""
        all_versions = self.get_all_versions()

        if not all_versions:
            return {
                'total_versions': 0,
                'message': 'No versions registered yet'
            }

        # Get metrics from all versions
        cognition_scores = [v.performance_metrics.get('cognition_score', 0) for v in all_versions]
        accuracies = [v.performance_metrics.get('accuracy', 0) for v in all_versions]

        baseline = self.get_baseline_version()

        stats = {
            'total_versions': len(all_versions),
            'active_version': self.active_version.version_id if self.active_version else None,
            'baseline_version': baseline.version_id if baseline else None,
            'avg_cognition_score': sum(cognition_scores) / len(cognition_scores),
            'best_cognition_score': max(cognition_scores),
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'best_accuracy': max(accuracies),
            'latest_version': all_versions[0].version_id if all_versions else None,
        }

        # Compare with baseline
        if baseline and self.active_version:
            baseline_cognition = baseline.performance_metrics.get('cognition_score', 0)
            active_cognition = self.active_version.performance_metrics.get('cognition_score', 0)

            stats['improvement_from_baseline'] = active_cognition - baseline_cognition

        return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DI Container Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_model_manager() -> ModelManager:
    """
    Get global model manager instance from DI container
    
    Returns:
        ModelManager instance (singleton)
    """
    try:
        from src.core.di import Container
        manager = Container.resolve("model_manager")
        if manager is not None:
            return manager
    except ImportError:
        pass
    
    # Fallback to manual singleton for backward compatibility
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
