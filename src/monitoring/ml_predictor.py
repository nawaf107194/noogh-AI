#!/usr/bin/env python3
"""
🤖 ML-Based VRAM Predictor
محرك التنبؤ بـ VRAM باستخدام Machine Learning

يتدرب على البيانات التاريخية للتنبؤ بـ VRAM المطلوب بدقة عالية.

Features:
- Random Forest Regressor
- Feature engineering from training sessions
- Model persistence (joblib)
- Confidence intervals
- Automatic retraining
"""

import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️  scikit-learn not available - ML predictions disabled")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VRAM Predictor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VRAMPredictor:
    """
    🤖 محرك التنبؤ بـ VRAM

    يتدرب على البيانات التاريخية للتنبؤ بـ VRAM المطلوب.
    """

    def __init__(
        self,
        model_path: str = "/home/noogh/projects/noogh_unified_system/logs/ml_models",
        verbose: bool = True
    ):
        """
        Initialize VRAM predictor.

        Args:
            model_path: Path to save/load trained models
            verbose: Enable verbose logging
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ML predictions. "
                "Install it with: pip install scikit-learn"
            )

        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Models
        self.vram_model: Optional[RandomForestRegressor] = None
        self.duration_model: Optional[RandomForestRegressor] = None

        # Encoders
        self.model_name_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()

        # Metadata
        self.is_trained = False
        self.training_date: Optional[datetime] = None
        self.n_training_samples = 0
        self.feature_names: List[str] = []

        # Performance metrics
        self.vram_mae = 0.0  # Mean Absolute Error
        self.vram_r2 = 0.0   # R² score
        self.duration_mae = 0.0
        self.duration_r2 = 0.0

        # Load existing model if available
        self._load_models()

        if self.verbose:
            logger.info("🤖 VRAM Predictor initialized")
            if self.is_trained:
                logger.info(f"   ✅ Loaded trained model ({self.n_training_samples} samples)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Feature Engineering
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_features(
        self,
        sessions: List[Dict[str, Any]],
        fit_encoders: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from training sessions.

        Args:
            sessions: List of training session dictionaries
            fit_encoders: Whether to fit label encoders

        Returns:
            (features, vram_targets, duration_targets)
        """
        features = []
        vram_targets = []
        duration_targets = []

        # Collect model names and devices for encoding
        if fit_encoders:
            model_names = [s.get('model_name', 'unknown') for s in sessions]
            devices = [s.get('device_used', 'gpu') for s in sessions]

            self.model_name_encoder.fit(model_names)
            self.device_encoder.fit(devices)

        for session in sessions:
            # Skip failed sessions
            if not session.get('success', False):
                continue

            # Extract features
            try:
                model_name = session.get('model_name', 'unknown')
                device = session.get('device_used', 'gpu')
                epochs = session.get('epochs', 10)
                ministers_count = session.get('ministers_count', 0)

                # Encode categorical features
                model_encoded = self.model_name_encoder.transform([model_name])[0]
                device_encoded = self.device_encoder.transform([device])[0]

                # Create feature vector
                feature_vector = [
                    model_encoded,        # Model type
                    device_encoded,       # CPU or GPU
                    epochs,               # Number of epochs
                    ministers_count,      # Number of paused ministers
                ]

                features.append(feature_vector)

                # Targets
                vram_targets.append(session.get('vram_peak_gb', 0.0))
                duration_targets.append(session.get('duration_seconds', 0.0))

            except Exception as e:
                logger.warning(f"Failed to extract features from session: {e}")
                continue

        if not features:
            raise ValueError("No valid training samples found")

        self.feature_names = [
            'model_type_encoded',
            'device_encoded',
            'epochs',
            'ministers_count'
        ]

        return (
            np.array(features),
            np.array(vram_targets),
            np.array(duration_targets)
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def train(
        self,
        sessions: List[Dict[str, Any]],
        test_size: float = 0.2,
        n_estimators: int = 100
    ) -> Dict[str, Any]:
        """
        Train the predictor on historical sessions.

        Args:
            sessions: List of training session dictionaries
            test_size: Fraction of data to use for testing
            n_estimators: Number of trees in random forest

        Returns:
            Training results with metrics
        """
        if self.verbose:
            logger.info(f"🎓 Training VRAM predictor on {len(sessions)} sessions...")

        # Extract features
        X, y_vram, y_duration = self._extract_features(sessions, fit_encoders=True)

        if len(X) < 10:
            raise ValueError(
                f"Not enough training data: {len(X)} samples. Need at least 10."
            )

        # Split data
        X_train, X_test, y_vram_train, y_vram_test = train_test_split(
            X, y_vram, test_size=test_size, random_state=42
        )
        _, _, y_dur_train, y_dur_test = train_test_split(
            X, y_duration, test_size=test_size, random_state=42
        )

        # Train VRAM model
        if self.verbose:
            logger.info("   Training VRAM prediction model...")

        self.vram_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.vram_model.fit(X_train, y_vram_train)

        # Evaluate VRAM model
        y_vram_pred = self.vram_model.predict(X_test)
        self.vram_mae = mean_absolute_error(y_vram_test, y_vram_pred)
        self.vram_r2 = r2_score(y_vram_test, y_vram_pred)

        # Train duration model
        if self.verbose:
            logger.info("   Training duration prediction model...")

        self.duration_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.duration_model.fit(X_train, y_dur_train)

        # Evaluate duration model
        y_dur_pred = self.duration_model.predict(X_test)
        self.duration_mae = mean_absolute_error(y_dur_test, y_dur_pred)
        self.duration_r2 = r2_score(y_dur_test, y_dur_pred)

        # Update metadata
        self.is_trained = True
        self.training_date = datetime.now()
        self.n_training_samples = len(X)

        # Save models
        self._save_models()

        results = {
            'success': True,
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'vram_prediction': {
                'mae_gb': round(self.vram_mae, 3),
                'r2_score': round(self.vram_r2, 3),
                'accuracy_percent': round((1 - self.vram_mae / y_vram.mean()) * 100, 2)
            },
            'duration_prediction': {
                'mae_seconds': round(self.duration_mae, 2),
                'r2_score': round(self.duration_r2, 3)
            },
            'training_date': self.training_date.isoformat()
        }

        if self.verbose:
            logger.info("✅ Training complete!")
            logger.info(f"   VRAM MAE: {self.vram_mae:.3f} GB")
            logger.info(f"   VRAM R²: {self.vram_r2:.3f}")
            logger.info(f"   Duration MAE: {self.duration_mae:.2f}s")

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Prediction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def predict_vram(
        self,
        model_name: str,
        epochs: int = 10,
        device: str = "gpu",
        ministers_count: int = 0,
        safety_margin: float = 1.1
    ) -> Dict[str, Any]:
        """
        Predict VRAM required for a training session.

        Args:
            model_name: Model name
            epochs: Number of epochs
            device: "cpu" or "gpu"
            ministers_count: Number of paused ministers
            safety_margin: Safety margin multiplier (default: 1.1 = +10%)

        Returns:
            {
                'predicted_vram_gb': float,
                'with_margin_gb': float,
                'confidence': float,
                'predicted_duration_seconds': float
            }
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Encode features
        try:
            model_encoded = self.model_name_encoder.transform([model_name])[0]
        except ValueError:
            # Unknown model - use mean encoding
            model_encoded = len(self.model_name_encoder.classes_) // 2
            if self.verbose:
                logger.warning(f"Unknown model '{model_name}', using default encoding")

        try:
            device_encoded = self.device_encoder.transform([device])[0]
        except ValueError:
            device_encoded = 1  # Default to GPU
            if self.verbose:
                logger.warning(f"Unknown device '{device}', using GPU encoding")

        # Create feature vector
        features = np.array([[
            model_encoded,
            device_encoded,
            epochs,
            ministers_count
        ]])

        # Predict VRAM
        vram_pred = self.vram_model.predict(features)[0]
        vram_with_margin = vram_pred * safety_margin

        # Predict duration
        duration_pred = self.duration_model.predict(features)[0]

        # Calculate confidence based on training R² score
        confidence = self.vram_r2

        result = {
            'predicted_vram_gb': round(vram_pred, 2),
            'with_margin_gb': round(vram_with_margin, 2),
            'safety_margin_percent': round((safety_margin - 1) * 100, 1),
            'confidence': round(confidence, 3),
            'predicted_duration_seconds': round(duration_pred, 1),
            'model_mae_gb': round(self.vram_mae, 3)
        }

        return result

    def predict_batch(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predict VRAM for multiple training sessions.

        Args:
            predictions: List of prediction requests

        Returns:
            List of prediction results
        """
        results = []

        for pred_req in predictions:
            try:
                result = self.predict_vram(**pred_req)
                results.append({
                    'success': True,
                    'request': pred_req,
                    'prediction': result
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'request': pred_req,
                    'error': str(e)
                })

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Model Persistence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save VRAM model
            joblib.dump(
                self.vram_model,
                self.model_path / "vram_predictor.joblib"
            )

            # Save duration model
            joblib.dump(
                self.duration_model,
                self.model_path / "duration_predictor.joblib"
            )

            # Save encoders
            joblib.dump(
                self.model_name_encoder,
                self.model_path / "model_name_encoder.joblib"
            )
            joblib.dump(
                self.device_encoder,
                self.model_path / "device_encoder.joblib"
            )

            # Save metadata
            metadata = {
                'is_trained': self.is_trained,
                'training_date': self.training_date.isoformat() if self.training_date else None,
                'n_training_samples': self.n_training_samples,
                'feature_names': self.feature_names,
                'vram_mae': self.vram_mae,
                'vram_r2': self.vram_r2,
                'duration_mae': self.duration_mae,
                'duration_r2': self.duration_r2
            }
            joblib.dump(metadata, self.model_path / "metadata.joblib")

            if self.verbose:
                logger.info(f"💾 Models saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def _load_models(self):
        """Load trained models from disk"""
        try:
            vram_model_file = self.model_path / "vram_predictor.joblib"
            duration_model_file = self.model_path / "duration_predictor.joblib"
            metadata_file = self.model_path / "metadata.joblib"

            if not vram_model_file.exists():
                return

            # Load models
            self.vram_model = joblib.load(vram_model_file)
            self.duration_model = joblib.load(duration_model_file)

            # Load encoders
            self.model_name_encoder = joblib.load(
                self.model_path / "model_name_encoder.joblib"
            )
            self.device_encoder = joblib.load(
                self.model_path / "device_encoder.joblib"
            )

            # Load metadata
            metadata = joblib.load(metadata_file)
            self.is_trained = metadata['is_trained']
            self.training_date = datetime.fromisoformat(metadata['training_date']) if metadata['training_date'] else None
            self.n_training_samples = metadata['n_training_samples']
            self.feature_names = metadata['feature_names']
            self.vram_mae = metadata['vram_mae']
            self.vram_r2 = metadata['vram_r2']
            self.duration_mae = metadata['duration_mae']
            self.duration_r2 = metadata['duration_r2']

        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to load models: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Statistics & Info
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_trained:
            return {
                'status': 'not_trained',
                'message': 'Model not trained yet'
            }

        return {
            'status': 'trained',
            'training_date': self.training_date.isoformat(),
            'n_training_samples': self.n_training_samples,
            'features': self.feature_names,
            'vram_prediction': {
                'mae_gb': round(self.vram_mae, 3),
                'r2_score': round(self.vram_r2, 3),
                'model_type': type(self.vram_model).__name__
            },
            'duration_prediction': {
                'mae_seconds': round(self.duration_mae, 2),
                'r2_score': round(self.duration_r2, 3),
                'model_type': type(self.duration_model).__name__
            },
            'known_models': list(self.model_name_encoder.classes_),
            'known_devices': list(self.device_encoder.classes_)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Instance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_predictor_instance = None

def get_vram_predictor(verbose: bool = False) -> Optional[VRAMPredictor]:
    """
    Get global VRAM predictor instance.

    Args:
        verbose: Enable verbose logging

    Returns:
        VRAMPredictor instance or None if sklearn not available
    """
    global _predictor_instance

    if not SKLEARN_AVAILABLE:
        return None

    if _predictor_instance is None:
        _predictor_instance = VRAMPredictor(verbose=verbose)

    return _predictor_instance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_predictor_from_evaluation_data(verbose: bool = True) -> Dict[str, Any]:
    """
    Train predictor using data from evaluation system.

    Args:
        verbose: Enable verbose logging

    Returns:
        Training results
    """
    from .self_evaluation import get_evaluation_system

    # Get evaluation system
    eval_system = get_evaluation_system(verbose=verbose)

    if not eval_system.recent_sessions:
        return {
            'success': False,
            'error': 'No training data available in evaluation system'
        }

    # Convert sessions to dicts
    sessions = [s.to_dict() for s in eval_system.recent_sessions]

    # Get predictor
    predictor = get_vram_predictor(verbose=verbose)

    if predictor is None:
        return {
            'success': False,
            'error': 'scikit-learn not available'
        }

    # Train
    try:
        results = predictor.train(sessions)
        return results
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
