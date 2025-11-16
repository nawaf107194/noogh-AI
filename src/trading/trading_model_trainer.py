#!/usr/bin/env python3
"""
🏋️ Trading Model Trainer - مدرب نماذج التداول
يدرب نماذج Deep Learning للتنبؤ بالصفقات باستخدام GPU
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import json

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PyTorch not available - trading trainer will not work")

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """تكوين التدريب"""
    model_type: str = "lstm"  # lstm, gru, transformer
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"  # Auto-detect GPU


@dataclass
class TrainingResult:
    """نتيجة التدريب"""
    status: str  # success, failed
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_loss: float
    val_loss: float
    epochs_trained: int
    model_path: str
    training_time: float
    best_epoch: int


class LSTMTradingModel(nn.Module):
    """نموذج LSTM للتنبؤ بالصفقات"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LSTMTradingModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism (optional but improves performance)
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 2)  # 2 classes: buy (1) or sell (0)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)

        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights shape: (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context shape: (batch, hidden_size)

        # Fully connected
        output = self.fc(context)
        # output shape: (batch, 2)

        return output


class TransformerTradingModel(nn.Module):
    """نموذج Transformer للتنبؤ بالصفقات"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.2
    ):
        super(TransformerTradingModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 100, hidden_size)  # max seq length = 100
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)
        # x shape: (batch, seq_len, hidden_size)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer
        transformer_out = self.transformer(x)
        # transformer_out shape: (batch, seq_len, hidden_size)

        # Use last token for classification
        last_token = transformer_out[:, -1, :]
        # last_token shape: (batch, hidden_size)

        # Output
        output = self.fc(last_token)
        # output shape: (batch, 2)

        return output


class TradingModelTrainer:
    """
    🏋️ مدرب نماذج التداول

    القدرات:
    - تدريب نماذج LSTM/Transformer على GPU
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Evaluation metrics
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        work_dir: str = "/home/noogh/projects/noogh_unified_system"
    ):
        self.config = config or TrainingConfig()
        self.work_dir = Path(work_dir)
        self.models_dir = self.work_dir / "models" / "trading"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.device = self.config.device

        # Stats
        self.models_trained = 0
        self.total_epochs = 0

        logger.info(f"🏋️ TradingModelTrainer initialized")
        logger.info(f"   Model type: {self.config.model_type}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Hidden size: {self.config.hidden_size}")
        logger.info(f"   Layers: {self.config.num_layers}")

        if not TORCH_AVAILABLE:
            logger.error("❌ PyTorch not available!")
            raise ImportError("PyTorch is required for TradingModelTrainer")

    def _create_model(self, input_size: int) -> nn.Module:
        """إنشاء النموذج"""

        if self.config.model_type == "lstm":
            model = LSTMTradingModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        elif self.config.model_type == "transformer":
            model = TransformerTradingModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        model = model.to(self.device)

        # عرض معلومات النموذج
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   ✅ Model created: {self.config.model_type.upper()}")
        logger.info(f"      Parameters: {num_params:,}")
        logger.info(f"      Device: {self.device}")

        return model

    def _prepare_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """تحضير DataLoaders"""

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        logger.info(f"   ✅ DataLoaders prepared:")
        logger.info(f"      Train batches: {len(train_loader)}")
        logger.info(f"      Val batches: {len(val_loader)}")

        return train_loader, val_loader

    async def train(
        self,
        dataset: Dict,
        validation_split: float = 0.2
    ) -> TrainingResult:
        """
        تدريب النموذج

        Args:
            dataset: Dataset من LiveMarketDataCollector
            validation_split: نسبة Validation

        Returns:
            TrainingResult
        """

        start_time = datetime.now()

        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"🏋️ Starting model training...")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        try:
            # Extract data
            X = dataset['X']
            y = dataset['y']
            symbol = dataset.get('symbol', 'unknown')

            logger.info(f"📊 Dataset: {symbol}")
            logger.info(f"   Samples: {len(X)}")
            logger.info(f"   Features: {X.shape[2]}")
            logger.info(f"   Sequence length: {X.shape[1]}")

            # Split train/val
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}")

            # Create model
            input_size = X.shape[2]
            self.model = self._create_model(input_size)

            # Prepare dataloaders
            train_loader, val_loader = self._prepare_dataloaders(
                X_train, y_train, X_val, y_val
            )

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )

            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )

            # Training loop
            logger.info(f"\n🚀 Training for {self.config.epochs} epochs...")

            best_val_loss = float('inf')
            best_epoch = 0
            patience_counter = 0

            train_losses = []
            val_losses = []

            for epoch in range(self.config.epochs):
                # Train
                train_loss = self._train_epoch(
                    self.model,
                    train_loader,
                    criterion,
                    optimizer
                )
                train_losses.append(train_loss)

                # Validate
                val_loss, val_metrics = self._validate_epoch(
                    self.model,
                    val_loader,
                    criterion
                )
                val_losses.append(val_loss)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Logging
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(
                        f"   Epoch {epoch+1}/{self.config.epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.2%}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0

                    # Save best model
                    self._save_checkpoint(self.model, symbol, epoch, val_loss)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"   ⏸️ Early stopping at epoch {epoch+1}")
                    break

            # Final evaluation
            logger.info(f"\n📊 Final evaluation...")
            _, final_metrics = self._validate_epoch(
                self.model,
                val_loader,
                criterion
            )

            # Save final model
            model_path = self._save_final_model(self.model, symbol)

            # Training time
            training_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"✅ Training complete!")
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"   Best epoch: {best_epoch}")
            logger.info(f"   Best val loss: {best_val_loss:.4f}")
            logger.info(f"   Final accuracy: {final_metrics['accuracy']:.2%}")
            logger.info(f"   Precision: {final_metrics['precision']:.2%}")
            logger.info(f"   Recall: {final_metrics['recall']:.2%}")
            logger.info(f"   F1 Score: {final_metrics['f1_score']:.2%}")
            logger.info(f"   Training time: {training_time:.1f}s")
            logger.info(f"   Model saved: {model_path}")
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            self.models_trained += 1
            self.total_epochs += epoch + 1

            return TrainingResult(
                status="success",
                accuracy=final_metrics['accuracy'],
                precision=final_metrics['precision'],
                recall=final_metrics['recall'],
                f1_score=final_metrics['f1_score'],
                train_loss=train_losses[-1],
                val_loss=val_losses[-1],
                epochs_trained=epoch + 1,
                model_path=str(model_path),
                training_time=training_time,
                best_epoch=best_epoch
            )

        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            return TrainingResult(
                status="failed",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                train_loss=0.0,
                val_loss=0.0,
                epochs_trained=0,
                model_path="",
                training_time=0.0,
                best_epoch=0
            )

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> float:
        """تدريب epoch واحد"""

        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, Dict]:
        """تقييم epoch واحد"""

        model.eval()
        total_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                # Predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = np.mean(all_preds == all_labels)

        # Precision, Recall, F1 for class 1 (buy signal)
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        return total_loss / len(val_loader), metrics

    def _save_checkpoint(
        self,
        model: nn.Module,
        symbol: str,
        epoch: int,
        val_loss: float
    ):
        """حفظ checkpoint"""
        symbol_clean = symbol.replace("/", "_")
        checkpoint_path = self.models_dir / f"{symbol_clean}_checkpoint_epoch{epoch+1}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'config': asdict(self.config)
        }, checkpoint_path)

    def _save_final_model(self, model: nn.Module, symbol: str) -> Path:
        """حفظ النموذج النهائي"""
        symbol_clean = symbol.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"{symbol_clean}_model_{timestamp}.pth"

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(self.config),
            'timestamp': timestamp,
            'symbol': symbol
        }, model_path)

        return model_path

    def load_model(self, model_path: str, input_size: int) -> nn.Module:
        """تحميل نموذج محفوظ"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = self._create_model(input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model

    def get_stats(self) -> Dict:
        """إحصائيات المدرب"""
        return {
            'models_trained': self.models_trained,
            'total_epochs': self.total_epochs,
            'device': self.device,
            'config': asdict(self.config)
        }


async def main():
    """اختبار المدرب"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("🏋️ Testing Trading Model Trainer")
    print("="*70 + "\n")

    # Create dummy dataset
    num_samples = 500
    seq_length = 60
    num_features = 14

    X = np.random.randn(num_samples, seq_length, num_features).astype(np.float32)
    y = np.random.randint(0, 2, num_samples)

    dataset = {
        'X': X,
        'y': y,
        'symbol': 'BTC/USDT',
        'num_samples': num_samples,
        'num_features': num_features
    }

    # Create trainer
    config = TrainingConfig(
        model_type="lstm",
        epochs=10,
        batch_size=32
    )

    trainer = TradingModelTrainer(config=config)

    # Train
    result = await trainer.train(dataset)

    # Display results
    print("\n" + "="*70)
    print("📊 Training Results:")
    print("="*70)
    print(f"Status: {result.status}")
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"Precision: {result.precision:.2%}")
    print(f"Recall: {result.recall:.2%}")
    print(f"F1 Score: {result.f1_score:.2%}")
    print(f"Epochs: {result.epochs_trained}")
    print(f"Time: {result.training_time:.1f}s")
    print(f"Model: {result.model_path}")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
