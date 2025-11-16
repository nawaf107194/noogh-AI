#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 MEGA BRAIN V5 HYBRID TRAINER
نظام تدريب ذكي يوزّع الأحمال بين CPU و GPU بشكل متوازن.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import psutil
import threading
from queue import Queue
from pathlib import Path

# 🔧 الإعدادات العامة
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("/home/noogh/models/megabrain_v5_checkpoints")
LOG_PATH = Path("/home/noogh/noogh_unified_system/core/brain/train_megabrain_v5_hybrid.log")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 🔹 إعداد التسجيل
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 🧠 نموذج الدماغ العصبي - التدريب على الـ GPU
class MegaBrainV5(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1024, num_layers=20):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ⚙️ وظيفة CPU لتوليد البيانات بشكل متوازي
def cpu_data_loader(queue: Queue, stop_event: threading.Event):
    logger.info("🧩 CPU Thread: بدأ توليد البيانات الخلفي...")
    while not stop_event.is_set():
        X = torch.randn(512, 1024)  # دفعة جديدة
        y = torch.randn(512, 1024)
        queue.put((X, y))
        time.sleep(0.2)
    logger.info("🧩 CPU Thread: تم إيقاف التوليد.")

# 🚀 وظيفة التدريب (GPU)
def train_hybrid():
    logger.info("🚀 بدء تدريب MEGA BRAIN V5 HYBRID MODE")
    logger.info("=" * 70)
    logger.info(f"🎮 الجهاز النشط: {DEVICE}")

    model = MegaBrainV5().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    data_queue = Queue(maxsize=5)
    stop_event = threading.Event()
    cpu_thread = threading.Thread(target=cpu_data_loader, args=(data_queue, stop_event))
    cpu_thread.start()

    epochs = 30
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        start = time.time()

        if not data_queue.empty():
            X_cpu, y_cpu = data_queue.get()
            X, y = X_cpu.to(DEVICE), y_cpu.to(DEVICE)
        else:
            X = torch.randn(512, 1024).to(DEVICE)
            y = torch.randn(512, 1024).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # 🔍 قياس الأداء
        duration = time.time() - start
        cpu_usage = psutil.cpu_percent()
        gpu_mem = torch.cuda.memory_allocated(DEVICE) / 1024**2
        gpu_total = torch.cuda.get_device_properties(DEVICE).total_memory / 1024**2
        gpu_usage = (gpu_mem / gpu_total) * 100

        logger.info(
            f"📊 Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f} | "
            f"CPU: {cpu_usage:.1f}% | GPU: {gpu_usage:.1f}% ({gpu_mem:.0f}/{gpu_total:.0f}MB) | "
            f"⏱️ {duration:.2f}s"
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
            logger.info(f"💾 نموذج محسّن تم حفظه (Loss={best_loss:.5f})")

    # إيقاف خيط الـ CPU
    stop_event.set()
    cpu_thread.join()

    logger.info("=" * 70)
    logger.info("✅ التدريب الهجين انتهى بنجاح!")
    logger.info(f"🧠 أفضل Loss: {best_loss:.5f}")
    logger.info(f"📁 النماذج: {SAVE_DIR}")
    logger.info("=" * 70)

# 🚀 نقطة البداية
if __name__ == "__main__":
    train_hybrid()
