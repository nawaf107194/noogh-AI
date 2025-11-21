# 👁️ Vision System Installation Guide

Quick guide to install and test the LLaVA vision system.

## 📦 Step 1: Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install vision dependencies
pip install transformers accelerate bitsandbytes pillow

# Verify installation
python3 -c "from transformers import LlavaNextProcessor; print('✅ Vision dependencies installed!')"
```

## 🧪 Step 2: Test Vision System

```bash
# Run vision test script
python3 scripts/test_vision.py
```

**Expected:** Model will download (~14GB) on first run, then analyze a sample chart.

## 🚀 Step 3: Launch Full System

```bash
# Start complete system with vision
./scripts/start_sovereign_system.sh
```

## 📊 VRAM Requirements

| Component | VRAM | Status |
|-----------|------|--------|
| Llama-3-8B (FP16) | ~3.5GB | ✅ Running |
| LLaVA-1.6 (4-bit) | ~4.0GB | ✅ New |
| **Total** | **~7.5GB** | ✅ Fits RTX 5070 |

## ⚡ Quick Test

Test vision directly in Python:

```python
from src.services.vision_service import VisionService

vision = VisionService()
result = vision.analyze_chart(
    "path/to/chart.png",
    "Is this bullish?"
)
print(result["analysis"])
```

## 🔍 Troubleshooting

**Out of VRAM?**
- Close other GPU apps
- Already using 4-bit quantization (most efficient)

**Slow download?**
- First run downloads ~14GB model
- Subsequent runs are instant (model cached)

**Import errors?**
- Verify dependencies: `pip list | grep transformers`
- Reinstall: `pip install --upgrade transformers`

---

✅ **Ready!** Vision system is now operational.
