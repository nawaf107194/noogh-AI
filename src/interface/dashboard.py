#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOOGH SOVEREIGN INTERFACE - Glass Citadel Dashboard
====================================================

Fixed: PIL.Image loading for chart display (no more network errors!)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import glob
from pathlib import Path
from datetime import datetime
from PIL import Image  # CRITICAL: For loading images properly
from streamlit_autorefresh import st_autorefresh
from streamlit_mic_recorder import mic_recorder

# Import System Core
# Add project root to path so 'src' module can be found
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.services.voice_service import VoiceService
from src.government.president import President
from src.core.settings import settings

# Configure Dashboard Logging
import logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "dashboard.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("dashboard")
logger.info("🚀 Dashboard startup initiated...")

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================

st.set_page_config(
    page_title="Noogh Sovereign Interface",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_path = Path(__file__).parent / "style.css"
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found - using default theme")

load_css()

# Auto-Refresh (disabled to prevent multiple model loads)
# st_autorefresh(interval=2000, key="system_pulse")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Lazy initialization with error handling
def get_voice_service():
    if "voice_service" not in st.session_state:
        try:
            logger.info("🎤 Initializing Voice Service...")
            st.session_state.voice_service = VoiceService()
            logger.info("✅ Voice Service ready")
        except Exception as e:
            logger.error(f"❌ Voice service failed: {e}", exc_info=True)
            st.error(f"Voice service unavailable: {e}")
            st.session_state.voice_service = None
    return st.session_state.voice_service

def get_president():
    if "president" not in st.session_state:
        try:
            logger.info("🎩 Initializing President...")
            st.session_state.president = President()
            logger.info("✅ President initialized successfully")
        except Exception as e:
            logger.error(f"❌ President initialization failed: {e}", exc_info=True)
            import traceback
            error_msg = f"President initialization failed: {e}\n{traceback.format_exc()}"
            st.error(error_msg)
            st.session_state.president = None
    return st.session_state.president

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

import json

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def read_system_logs(lines=50):
    """Reads the last N lines of the API log."""
    log_path = settings.logs_dir / "api.log"
    try:
        if not log_path.exists():
            return "⏳ Waiting for system logs..."

        # Efficiently read last N lines
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"❌ Error reading logs: {e}"

def get_trading_performance():
    """
    Reads paper_ledger.json and calculates performance metrics.
    """
    ledger_path = settings.data_dir / "paper_ledger.json"
    try:
        if not ledger_path.exists():
            return None, None

        with open(ledger_path, "r") as f:
            data = json.load(f)
        
        trades = data.get("trades", [])
        stats = data.get("stats", {})
        
        # Calculate Win Rate
        total_closed = stats.get("total_trades", 0) # Simplified
        wins = stats.get("profitable_trades", 0)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        
        # Calculate Portfolio Value (Cash + Unrealized PnL would be ideal, but for now just Cash)
        balances = data.get("balances", {})
        total_value = balances.get("spot_usdt", 0) + balances.get("futures_usdt", 0)
        
        # Create DataFrame
        if trades:
            df = pd.DataFrame(trades)
            # Sort by timestamp descending
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            # Select key columns
            df = df[['timestamp', 'market_type', 'symbol', 'side', 'entry_price', 'status', 'ai_confidence']]
        else:
            df = pd.DataFrame()
            
        return {
            "total_value": total_value,
            "win_rate": win_rate,
            "active_trades": len([t for t in trades if t['status'] == 'OPEN']),
            "total_pnl": stats.get("total_pnl", 0.0)
        }, df
        
    except Exception as e:
        st.error(f"Error reading ledger: {e}")
        return None, None

def get_latest_hunter_vision():
    """
    Finds the newest chart image generated by Hunter.

    Returns:
        (image_object, caption) or (None, None)
    """
    charts_dir = settings.data_dir / "charts"

    try:
        if not charts_dir.exists():
            return None, None

        files = list(charts_dir.glob("*.png"))
        if not files:
            return None, None

        latest_file = max(files, key=os.path.getctime)
        caption = f"Analysis: {latest_file.stem.replace('_', ' ').title()}"

        # CRITICAL FIX: Load image with PIL instead of passing path string
        image = Image.open(latest_file)
        return image, caption

    except Exception as e:
        st.error(f"Error loading chart: {e}")
        return None, None

def toggle_kill_switch():
    """Toggles the emergency stop file."""
    stop_file = settings.data_dir / "EMERGENCY_STOP"
    if stop_file.exists():
        stop_file.unlink()
        st.toast("✅ System Resumed", icon="🟢")
    else:
        stop_file.touch()
        st.toast("🛑 EMERGENCY STOP ACTIVATED", icon="🚨")

def get_system_telemetry():
    """
    Get REAL system telemetry from HealthMinister.

    Returns:
        Dict with GPU, CPU, Memory, and Disk stats or None if unavailable
    """
    try:
        # Only try to get telemetry if President is already initialized
        # This prevents blocking the dashboard on startup
        if "president" not in st.session_state:
            logger.info("ℹ️ President not yet initialized - skipping telemetry")
            return None

        president = st.session_state.president
        if not president or 'health' not in president.cabinet:
            return None

        health_minister = president.cabinet['health']
        vitals = health_minister.check_vital_signs()

        return vitals

    except Exception as e:
        logger.error(f"⚠️ Error getting telemetry: {e}")
        return None

def save_minister_state(state):
    """Saves minister state to config file."""
    config_path = project_root / "config" / "minister_state.json"
    try:
        with open(config_path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save minister state: {e}")

def load_minister_state():
    """Loads minister state from config file."""
    config_path = project_root / "config" / "minister_state.json"
    default_state = {
        "finance": True, "health": True, "development": True, 
        "security": True, "education": True, "foreign": True, "communication": True
    }
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return default_state
    except Exception:
        return default_state

# ==============================================================================
# SIDEBAR: THE CONTROLLER
# ==============================================================================

with st.sidebar:
    st.title("🦅 NOOGH OS")
    st.caption(f"v{settings.version} | {settings.cognition_level}")

    # President Status & Initialization
    if "president" not in st.session_state:
        st.warning("🎩 President not loaded")
        if st.button("⚡ Initialize President", type="primary", use_container_width=True):
            with st.spinner("🎩 Loading President & Ministers..."):
                get_president()
                st.rerun()
    else:
        if st.session_state.president:
            st.success("🎩 President Active")
        else:
            st.error("❌ President Failed to Load")

    st.markdown("---")

    # 1. Voice Command
    st.subheader("🎙️ Command")
    c1, c2 = st.columns([3, 1])
    with c1:
        audio = mic_recorder(
            start_prompt="🎤 Speak",
            stop_prompt="🛑 Stop",
            key='recorder',
            format="wav",
            use_container_width=True
        )
    with c2:
        tts_enabled = st.toggle("TTS", value=True, label_visibility="collapsed")
        st.caption("TTS")

    # Process Voice Input
    if audio:
        # Only process if President is initialized
        if "president" not in st.session_state or not st.session_state.president:
            st.warning("⚠️ Please initialize President first")
        else:
            voice_service = get_voice_service()
            president = st.session_state.president
            if voice_service and president:
                # Transcribe
                text = voice_service.transcribe(audio['bytes'])
                if text:
                    st.session_state.messages.append({"role": "user", "content": text})
                    # Process with President
                    response = president.process_request(text)
                    ai_reply = response.get("response", "Processing...")
                    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                    # Speak
                    if tts_enabled:
                        voice_service.speak(ai_reply)

    st.markdown("---")

    # 2. Kill Switch
    stop_file = settings.data_dir / "EMERGENCY_STOP"
    is_stopped = stop_file.exists()

    if is_stopped:
        st.error("🛑 SYSTEM HALTED")
        if st.button("RESUME OPERATIONS", key="resume_btn", use_container_width=True):
            toggle_kill_switch()
            st.rerun()
    else:
        st.success("🟢 SYSTEM ACTIVE")
        if st.button("🛑 KILL SWITCH", key="kill_btn", type="primary", use_container_width=True):
            toggle_kill_switch()
            st.rerun()

    st.markdown("---")

    # 3. Minister Control Center
    st.subheader("🏛️ Cabinet Control")
    
    current_state = load_minister_state()
    new_state = current_state.copy()
    state_changed = False
    
    for minister, is_active in current_state.items():
        # Use a unique key for each toggle
        toggle_val = st.toggle(
            f"{minister.title()} Minister", 
            value=is_active,
            key=f"toggle_{minister}"
        )
        if toggle_val != is_active:
            new_state[minister] = toggle_val
            state_changed = True
            
    if state_changed:
        save_minister_state(new_state)
        st.toast("Configuration Saved", icon="💾")
        # Optional: st.rerun() if immediate update needed

# ==============================================================================
# MAIN DASHBOARD: GLASS CITADEL LAYOUT
# ==============================================================================

# 1. TOP ROW: VITAL SIGNS (4 COLUMNS)
# ------------------------------------------------------------------------------
st.markdown("### 📡 System Telemetry")
k1, k2, k3, k4 = st.columns(4)

# Get REAL telemetry from HealthMinister (with safe error handling)
with st.spinner("📡 Establishing Neural Link..."):
    try:
        vitals = get_system_telemetry()
    except Exception as e:
        st.warning(f"⚠️ Telemetry error: {e}")
        vitals = None

# Get Trading Performance
perf_metrics, trades_df = get_trading_performance()

if vitals:
    # Extract GPU data
    gpu_data = vitals.get('gpu', {})
    gpu_temp = f"{gpu_data.get('temperature_c', 0)}°C" if gpu_data.get('status') != 'unavailable' else "N/A"
    vram_used = gpu_data.get('vram_used_mb', 0)
    vram_total = gpu_data.get('vram_total_mb', 1)
    vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
    vram_usage = f"{vram_used / 1024:.1f} GB" if gpu_data.get('status') != 'unavailable' else "N/A"
    vram_delta = f"{vram_percent:.1f}%"

    # GPU Status
    gpu_status = gpu_data.get('status', 'unknown')
    gpu_delta_text = gpu_status.upper() if gpu_status in ['warning', 'critical'] else ""

else:
    # Fallback to safe defaults if HealthMinister unavailable
    gpu_temp = "N/A"
    vram_usage = "N/A"
    vram_delta = "Offline"
    gpu_delta_text = "OFFLINE"

# Trading Metrics
if perf_metrics:
    paper_pnl = f"${perf_metrics['total_pnl']:,.2f}"
    portfolio_val = f"${perf_metrics['total_value']:,.2f}"
    win_rate_txt = f"{perf_metrics['win_rate']:.1f}%"
else:
    paper_pnl = "N/A"
    portfolio_val = "N/A"
    win_rate_txt = "N/A"

# Threat Level (TODO: Get from SecurityMinister)
threat_level = "LOW"

# Display metrics with REAL data
with k1:
    st.metric("GPU Temp", gpu_temp, gpu_delta_text if vitals else "Offline")

with k2:
    st.metric("VRAM Usage", vram_usage, vram_delta if vitals and gpu_temp != "N/A" else "N/A")

with k3:
    st.metric("Portfolio Value", portfolio_val, paper_pnl)

with k4:
    st.metric("Threat Level", threat_level, "Stable")

st.markdown("<br>", unsafe_allow_html=True)

# 2. MIDDLE SECTION: TABS FOR DIFFERENT VIEWS
# ------------------------------------------------------------------------------
tab_main, tab_market, tab_logs = st.tabs(["🧠 Neural Link", "📈 Market Data", "📜 System Logs"])

with tab_main:
    row2_1, row2_2 = st.columns([1, 1])

    # LEFT: CHAT INTERFACE
    with row2_1:
        st.markdown("### Chat Stream")
        chat_container = st.container(height=400)

        with chat_container:
            for msg in st.session_state.messages:
                role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
                st.markdown(f"""
                <div class="chat-message {role_class}">
                    <b>{msg['role'].upper()}:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True)

        # Text Input (Fallback for Voice)
        # Check if President is loaded before allowing chat
        if "president" not in st.session_state or not st.session_state.president:
            st.info("💡 Please initialize President in the sidebar to enable chat")
        else:
            if prompt := st.chat_input("Enter command..."):
                president = st.session_state.president
                voice_service = get_voice_service()
                if president:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    # Process with President
                    response = president.process_request(prompt)
                    ai_reply = response.get("response", "Processing...")
                    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                    if tts_enabled and voice_service:
                        voice_service.speak(ai_reply)
                    st.rerun()

    # RIGHT: VISUAL CORTEX (Chart Viewer)
    with row2_2:
        st.markdown("### 👁️ Visual Cortex")

        # CRITICAL FIX: Use PIL.Image object instead of path string
        try:
            img_object, caption = get_latest_hunter_vision()

            if img_object:
                st.image(img_object, caption=caption, use_container_width=True)
            else:
                st.info("No visual data available. Hunter is scanning markets...")
                # Placeholder for aesthetic
                st.markdown("""
                <div style="height: 350px; border: 1px dashed #333; display: flex; align-items: center; justify-content: center; color: #555;">
                    ⏳ WAITING FOR VISUAL INPUT...
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Visual Cortex Error: {e}")
            st.markdown("""
            <div style="height: 350px; border: 1px solid #f44336; display: flex; align-items: center; justify-content: center; color: #f44336;">
                ⚠️ CHART LOADING ERROR
            </div>
            """, unsafe_allow_html=True)

with tab_market:
    st.markdown("### 📊 Live Trading Data")
    if trades_df is not None and not trades_df.empty:
        st.dataframe(
            trades_df,
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm"),
                "entry_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "ai_confidence": st.column_config.TextColumn("Confidence"),
            },
            hide_index=True
        )
    else:
        st.info("No trades recorded yet.")

with tab_logs:
    st.markdown("### 📜 Stream of Consciousness")
    logs = read_system_logs(lines=100)
    st.code(logs, language="text")
