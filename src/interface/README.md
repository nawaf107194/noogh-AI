# Noogh Interface Module

This module contains the Streamlit-based dashboard interface for the Noogh Sovereign System.

## Files

- `dashboard.py` - Main Streamlit application
- `style.css` - Professional dark glass design system

## Usage

Launch the dashboard:

```bash
./scripts/launch_dashboard.sh
```

Or manually:

```bash
streamlit run src/interface/dashboard.py --server.port 8501 --theme.base "dark"
```

## Features

- **Real-time System Monitoring**: GPU, CPU, RAM stats updated every 2 seconds
- **Cabinet Status**: Live status of all 7 Ministers
- **Command Center**: Chat interface with AI Government
- **Market Hunter**: Paper trading analytics with interactive charts
- **Security Center**: System logs and security monitoring

## Design

Professional Bloomberg Terminal-inspired dark theme with:

- Dark grey background (#1e1e1e)
- Glass-morphism cards
- Professional blue accent (#2962FF)
- Inter/Roboto typography
- Responsive layout
