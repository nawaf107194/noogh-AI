#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Telemetry Fix
=======================

This file contains the FIXED code to replace the hardcoded mock data
in dashboard.py with REAL telemetry from HealthMinister.

APPLY THIS FIX TO: src/interface/dashboard.py
"""

# ============================================================================
# STEP 1: Add helper function to get real telemetry
# ============================================================================
# ADD THIS AFTER LINE 108 (after get_latest_hunter_vision function)

def get_system_telemetry():
    """
    Get REAL system telemetry from HealthMinister.

    Returns:
        Dict with GPU, CPU, Memory, and Disk stats
    """
    try:
        president = get_president()
        if not president or 'health' not in president.cabinet:
            return None

        health_minister = president.cabinet['health']
        vitals = health_minister.check_vital_signs()

        return vitals

    except Exception as e:
        import logging
        logging.error(f"Error getting telemetry: {e}")
        return None


# ============================================================================
# STEP 2: Replace the hardcoded metrics section
# ============================================================================
# REPLACE LINES 199-228 WITH THIS:

"""
# 1. TOP ROW: VITAL SIGNS (REAL DATA)
# ------------------------------------------------------------------------------
st.markdown("### 📡 System Telemetry")
k1, k2, k3, k4 = st.columns(4)

# Get REAL telemetry from HealthMinister
vitals = get_system_telemetry()

if vitals:
    # Extract GPU data
    gpu_data = vitals.get('gpu', {})
    gpu_temp = f"{gpu_data.get('temperature_c', 0)}°C"
    vram_used = gpu_data.get('vram_used_mb', 0)
    vram_total = gpu_data.get('vram_total_mb', 1)
    vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
    vram_usage = f"{vram_used / 1024:.1f} GB"
    vram_delta = f"{vram_percent:.1f}%"

    # Extract CPU data
    cpu_data = vitals.get('cpu', {})
    cpu_percent = cpu_data.get('percent', 0)

    # Extract Memory data
    memory_data = vitals.get('memory', {})
    memory_percent = memory_data.get('percent', 0)

    # GPU Status based on temperature
    gpu_status = gpu_data.get('status', 'unknown')
    gpu_delta_color = "normal" if gpu_status == "healthy" else "inverse"

else:
    # Fallback to safe defaults if HealthMinister unavailable
    gpu_temp = "N/A"
    vram_usage = "N/A"
    vram_delta = "N/A"
    cpu_percent = 0
    memory_percent = 0
    gpu_delta_color = "off"

# Paper PnL (TODO: Get from FinanceMinister's ledger)
paper_pnl = "+$1,240"  # Placeholder until ledger integration

# Threat Level (TODO: Get from SecurityMinister)
threat_level = "LOW"

# Display metrics
with k1:
    st.markdown('<div class="metric-card metric-card-health">', unsafe_allow_html=True)
    if vitals:
        st.metric("GPU Temp", gpu_temp, delta=gpu_data.get('status', '').upper(), delta_color=gpu_delta_color)
    else:
        st.metric("GPU Temp", gpu_temp, "Unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="metric-card metric-card-health">', unsafe_allow_html=True)
    if vitals:
        st.metric("VRAM Usage", vram_usage, vram_delta)
    else:
        st.metric("VRAM Usage", vram_usage, "Unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="metric-card metric-card-finance">', unsafe_allow_html=True)
    st.metric("Paper PnL", paper_pnl, "+12%")
    st.markdown('</div>', unsafe_allow_html=True)

with k4:
    st.markdown('<div class="metric-card metric-card-alert">', unsafe_allow_html=True)
    st.metric("Threat Level", threat_level, "Stable")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
"""


# ============================================================================
# ALTERNATIVE: Minimal Fix (Just replace the hardcoded values)
# ============================================================================
# If you want a quicker fix, just replace lines 204-208:

"""
# Get real telemetry
vitals = get_system_telemetry()
if vitals:
    gpu_data = vitals.get('gpu', {})
    gpu_temp = f"{gpu_data.get('temperature_c', 0)}°C"
    vram_used_gb = gpu_data.get('vram_used_mb', 0) / 1024
    vram_usage = f"{vram_used_gb:.1f} GB"

    # TODO: Get real paper PnL from FinanceMinister
    paper_pnl = "+$1,240"

    # TODO: Get real threat level from SecurityMinister
    threat_level = "LOW"
else:
    # Fallback
    gpu_temp = "N/A"
    vram_usage = "N/A"
    paper_pnl = "N/A"
    threat_level = "N/A"
"""


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
1. Open: src/interface/dashboard.py

2. Add the get_system_telemetry() function after line 108

3. Replace lines 199-228 with the new telemetry section above

4. Save and restart dashboard:
   ./scripts/start_sovereign_system.sh

5. Verify:
   - GPU temp should show REAL temperature
   - VRAM should show ACTUAL usage
   - Metrics should update when you refresh
"""


# ============================================================================
# EXPECTED RESULT
# ============================================================================
"""
Before:
┌─────────────────┐
│ GPU Temp: 42°C  │  ← Hardcoded
│ VRAM: 3.2 GB    │  ← Hardcoded
└─────────────────┘

After:
┌─────────────────┐
│ GPU Temp: 67°C  │  ← Real from pynvml
│ VRAM: 7.8 GB    │  ← Real from GPU
└─────────────────┘
"""
