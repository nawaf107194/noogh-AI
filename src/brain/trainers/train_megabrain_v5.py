#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎓 MegaBrain V5 Training System
================================

نظام تدريب متقدم لـ MegaBrain V5 مع:
- GPU acceleration
- Smart monitoring
- Auto checkpointing
- Adaptive learning rate
- Real-time metrics
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain.run_brain_v5_adaptive import (
    AdaptiveMegaBrainV5,
    AdvancedGPUProbe,
    IntelligentResourceMonitor,
    SmartAutoPauseGuard,
    GradualWorkloadManager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("training_megabrain_v5.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("MEGABRAIN-TRAINER")


class SyntheticBrainDataset(Dataset):
    """
    مجموعة بيانات صناعية لتدريب الدماغ

    في المستقبل، يمكن استبدالها ببيانات حقيقية من:
    - Knowledge Base
    - User interactions
    - Self-learning patterns
    """

    def __init__(self, num_samples: int = 1000, input_size: int = 512, output_size: int = 512):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size

        log.info(f"📦 إنشاء مجموعة بيانات: {num_samples} عينة")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic input-output pairs
        # Pattern: y = f(x) where f is a complex transformation
        x = torch.randn(self.input_size)

        # Create a simple pattern to learn: scaled and transformed
        y = torch.tanh(x * 0.5 + 0.1)

        return {
            'input': x,
            'target': y
        }


class MegaBrainV5Trainer:
    """
    🎓 مدرب MegaBrain V5

    Full training pipeline with monitoring and optimization
    """

    def __init__(
        self,
        model: AdaptiveMegaBrainV5,
        gpu_probe: AdvancedGPUProbe,
        learning_rate: float = 0.0001,
        checkpoint_dir: str = "/home/noogh/brain_checkpoints"
    ):
        self.model = model
        self.gpu = gpu_probe
        self.device = gpu_probe.device

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # Monitoring
        self.monitor = IntelligentResourceMonitor(gpu_probe)
        self.guard = SmartAutoPauseGuard(self.monitor)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': [],
            'timestamps': []
        }

        log.info(f"🎓 MegaBrain V5 Trainer initialized")
        log.info(f"   Device: {self.device}")
        log.info(f"   Learning Rate: {learning_rate}")
        log.info(f"   Checkpoint Dir: {checkpoint_dir}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """تدريب epoch واحد"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Check resources before processing
            self.guard.pause_if_needed()

            # Move data to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, guard=self.guard)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 10 == 0:
                log.debug(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f} | {self.monitor.get_report()}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """تقييم النموذج"""
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
        epochs: int = 20,
        early_stopping_patience: int = 5
    ) -> Dict:
        """
        التدريب الكامل

        Args:
            train_loader: DataLoader للتدريب
            val_loader: DataLoader للتقييم
            epochs: عدد الـ epochs
            early_stopping_patience: صبر Early Stopping

        Returns:
            Training results
        """
        log.info("=" * 70)
        log.info("🚀 بدء تدريب MegaBrain V5")
        log.info("=" * 70)

        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['epochs'].append(epoch)
            self.history['timestamps'].append(datetime.now().isoformat())

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)

                # Update scheduler
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    patience_counter += 1

                # Check early stopping
                if patience_counter >= early_stopping_patience:
                    log.info(f"⏹️  Early stopping at epoch {epoch}")
                    break

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Log epoch results
            epoch_time = time.time() - epoch_start
            val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"

            log.info(
                f"📊 Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_str} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s | "
                f"{self.monitor.get_report()}"
            )

            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_loss)

        total_time = time.time() - start_time

        # Final results
        results = {
            'total_epochs': epoch,
            'best_val_loss': best_val_loss if val_loader else None,
            'final_train_loss': train_loss,
            'training_time': total_time,
            'history': self.history,
            'guard_stats': self.guard.get_stats(),
            'model_stats': self.model.get_performance_stats()
        }

        # Save training history
        self.save_history()

        log.info("=" * 70)
        log.info("✅ التدريب اكتمل!")
        log.info(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        log.info(f"   Best Val Loss: {best_val_loss:.4f}")
        log.info(f"   Pauses: {self.guard.pause_count}")
        log.info(f"   Peak VRAM: {self.gpu.peak_memory_used:.0f} MB")
        log.info("=" * 70)

        return results

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: Optional[float],
        is_best: bool = False
    ):
        """حفظ checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            log.info(f"💾 Saved best model: {path}")
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            log.info(f"💾 Saved checkpoint: {path}")

    def save_history(self):
        """حفظ تاريخ التدريب"""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        log.info(f"📝 Saved training history: {history_path}")


def main():
    """الدالة الرئيسية للتدريب"""

    log.info("🧠 MegaBrain V5 Training System")
    log.info("=" * 70)

    # Setup
    gpu = AdvancedGPUProbe(0)

    # Create model
    model = AdaptiveMegaBrainV5(
        input_size=512,
        hidden_size=1024,
        layers=8,
        output_size=512,
        efficient=True
    )

    log.info(f"📦 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create datasets
    train_dataset = SyntheticBrainDataset(num_samples=2000)
    val_dataset = SyntheticBrainDataset(num_samples=500)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Create trainer
    trainer = MegaBrainV5Trainer(
        model=model,
        gpu_probe=gpu,
        learning_rate=0.0001
    )

    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        early_stopping_patience=5
    )

    # Print final stats
    log.info("\n" + "=" * 70)
    log.info("📊 Final Statistics:")
    log.info("=" * 70)
    for key, value in results.items():
        if key not in ['history', 'guard_stats', 'model_stats']:
            log.info(f"   {key}: {value}")

    log.info("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
