#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Neural Brain Trainer - نظام تدريب الدماغ العصبي
================================================

Complete training system for Neural Brain v4.0 (32K neurons)

Features:
- Full neural network training with backpropagation
- GPU/CPU support
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Training metrics & logging

Author: Noogh AI Team
Version: 1.0.0
Date: 2025-10-25
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone
import json
import time

logger = logging.getLogger(__name__)


class BrainTrainingDataset(Dataset):
    """
    Dataset للتدريب العصبوني

    يأخذ patterns من Self-Learning Loop
    ويحولها إلى training examples
    """

    def __init__(self, patterns: List[Dict[str, Any]], max_length: int = 512):
        """
        Initialize dataset

        Args:
            patterns: قائمة الأنماط من Self-Learning
            max_length: أقصى طول للـ sequence
        """
        self.patterns = patterns
        self.max_length = max_length
        self.data = self._process_patterns()

    def _process_patterns(self) -> List[Dict[str, torch.Tensor]]:
        """تحويل الأنماط إلى tensors"""
        processed = []

        for pattern in self.patterns:
            try:
                # Extract features from pattern
                input_features = self._extract_features(pattern)
                target = self._extract_target(pattern)

                if input_features is not None and target is not None:
                    processed.append({
                        'input': torch.FloatTensor(input_features),
                        'target': torch.FloatTensor(target)
                    })
            except Exception as e:
                logger.warning(f"Error processing pattern: {e}")
                continue

        return processed

    def _extract_features(self, pattern: Dict) -> Optional[List[float]]:
        """
        استخراج features من pattern

        مثال: [confidence, success_rate, occurrences, ...]
        """
        try:
            features = [
                pattern.get('confidence', 0.0),
                pattern.get('success_rate', 0.0),
                float(pattern.get('occurrences', 0)),
                # Add more features as needed
            ]

            # Normalize occurrences
            if features[2] > 0:
                features[2] = min(features[2] / 100.0, 1.0)

            return features
        except Exception as e:
            return None

    def _extract_target(self, pattern: Dict) -> Optional[List[float]]:
        """استخراج target من pattern"""
        try:
            # Target: should we apply this pattern? (0 or 1)
            should_apply = 1.0 if pattern.get('success_rate', 0) > 0.7 else 0.0
            return [should_apply]
        except Exception as e:
            return None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class NeuralBrainTrainer:
    """
    🧠 مدرب الدماغ العصبي

    Full training system for Neural Brain v4.0
    """

    def __init__(
        self,
        brain_model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01
    ):
        """
        Initialize trainer

        Args:
            brain_model: النموذج العصبوني (Neural Brain v4.0)
            device: "cpu" or "cuda"
            learning_rate: معدل التعلم
            weight_decay: L2 regularization
        """
        self.model = brain_model
        self.device = torch.device(device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        logger.info(f"🧠 NeuralBrainTrainer initialized (device: {device})")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        تدريب epoch واحد

        Args:
            train_loader: DataLoader للتدريب
            epoch: رقم الـ epoch

        Returns:
            Average loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        تقييم النموذج على validation set

        Args:
            val_loader: DataLoader للتقييم

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: int = 5,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        التدريب الكامل

        Args:
            train_loader: DataLoader للتدريب
            val_loader: DataLoader للتقييم (اختياري)
            epochs: عدد الـ epochs
            early_stopping_patience: صبر Early Stopping
            checkpoint_dir: مجلد حفظ الـ checkpoints

        Returns:
            Training results dictionary
        """
        logger.info("=" * 70)
        logger.info("🚀 بدء التدريب العصبوني")
        logger.info("=" * 70)

        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0

        # Create checkpoint directory
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    if checkpoint_dir:
                        self.save_checkpoint(
                            checkpoint_dir,
                            epoch,
                            val_loss,
                            is_best=True
                        )
                else:
                    patience_counter += 1

                # Check early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"⏹️  Early stopping triggered at epoch {epoch} "
                        f"(patience: {early_stopping_patience})"
                    )
                    break
            else:
                val_loss = None

            # Log epoch results
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss_str}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )

        total_time = time.time() - start_time

        results = {
            'total_epochs': epoch,
            'best_val_loss': best_val_loss if val_loader else None,
            'final_train_loss': train_loss,
            'training_time': total_time,
            'history': self.history
        }

        logger.info("=" * 70)
        logger.info("✅ التدريب اكتمل!")
        logger.info(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        logger.info(f"   Best val loss: {best_val_loss:.4f}")
        logger.info("=" * 70)

        return results

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """
        حفظ checkpoint

        Args:
            checkpoint_dir: مجلد الحفظ
            epoch: رقم الـ epoch
            val_loss: validation loss
            is_best: هل هذا أفضل نموذج؟
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path / filename)

        # Save best model
        if is_best:
            best_filename = "best_model.pt"
            torch.save(checkpoint, checkpoint_path / best_filename)
            logger.info(f"💾 Saved best model: {best_filename}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        تحميل checkpoint

        Args:
            checkpoint_path: مسار الـ checkpoint

        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        logger.info(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint

    def get_training_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التدريب"""
        return {
            'total_epochs': len(self.history['train_loss']),
            'best_train_loss': min(self.history['train_loss']) if self.history['train_loss'] else None,
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
            'final_learning_rate': self.history['learning_rates'][-1] if self.history['learning_rates'] else None,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def create_brain_trainer(
    brain_model: nn.Module,
    device: str = "cpu",
    learning_rate: float = 0.001
) -> NeuralBrainTrainer:
    """
    Factory function to create NeuralBrainTrainer

    Args:
        brain_model: Neural Brain model
        device: "cpu" or "cuda"
        learning_rate: Learning rate

    Returns:
        NeuralBrainTrainer instance
    """
    return NeuralBrainTrainer(
        brain_model=brain_model,
        device=device,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    # Test the trainer
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 70)
    print("🧪 Testing NeuralBrainTrainer")
    print("=" * 70 + "\n")

    # Create a simple test model
    test_model = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    # Create trainer
    trainer = create_brain_trainer(
        brain_model=test_model,
        device="cpu",
        learning_rate=0.001
    )

    # Create dummy data
    dummy_patterns = [
        {
            'confidence': 0.8,
            'success_rate': 0.9,
            'occurrences': 10
        }
        for _ in range(100)
    ]

    dataset = BrainTrainingDataset(dummy_patterns)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Train
    results = trainer.train(
        train_loader=train_loader,
        epochs=5
    )

    # Stats
    print("\n" + "=" * 70)
    print("📊 Training Statistics")
    print("=" * 70)

    stats = trainer.get_training_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ Test completed!")
