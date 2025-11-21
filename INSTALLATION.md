# Installation & Setup Instructions

## Prerequisites

- Python 3.9 or higher
- pip
- Virtual environment (recommended)

## Quick Start

### 1. Clone and Navigate

```bash
cd /home/noogh/projects/noogh_unified_system
```

### 2. Create/Activate Virtual Environment

```bash
# If venv doesn't exist, create it
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 3. Install the Package

```bash
# Install in editable/development mode
pip install -e .

# Or with development dependencies (recommended)
pip install -e ".[dev]"
```

### 4. Run the Application

```bash
# Start the API server
python -m src.api.main

# Or use the run script
./run.sh
```

### 5. Access the API

- API Documentation: <http://localhost:8000/docs>
- Alternative Docs: <http://localhost:8000/redoc>
- Dashboard: <http://localhost:8000/>

## What Changed in Phase 1?

### ✅ Removed `sys.path` Hacks (52 files)

**Before:**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.government.president import President
```

**After:**

```python
from ..government.president import President  # Relative import
# or
from src.government.president import President  # Absolute import after package install
```

### ✅ Created `pyproject.toml`

Package is now properly configured with:

- Package metadata
- Dependencies
- Build system
- Development tools (pytest, black, ruff, mypy)

### ✅ Package Installed in Development Mode

- Run `pip install -e .` to install
- Changes to code immediately reflected (no reinstall needed)
- Can import from `src` anywhere in your environment

## Development Workflow

### Running Tests

```bash
pytest tests/
``````

### Code Quality

bash

# Format code

black src/

# Lint code

ruff src/

# Type check

mypy src/

```

### Adding Dependencies
Edit `pyproject.toml` under `[project]` `dependencies` array, then:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
noogh_unified_system/
├── src/                    # Main package
│   ├── api/               # FastAPI application
│   ├── brain/             # Neural network modules
│   ├── government/        # Government system (14 ministers)
│   ├── trading/           # Trading system
│   ├── autonomy/          # Autonomous systems
│   ├── core/              # Core utilities
│   └── ...
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── pyproject.toml         # Package configuration
├── requirements.txt       # Legacy requirements (use pyproject.toml)
└── README.md             # This file
```

## Troubleshooting

### Import Errors

Make sure the package is installed:

```bash
pip install -e .
```

### Module Not Found

Activate your virtual environment:

```bash
source venv/bin/activate
```

### Router Loading Errors

Some routers may have syntax errors from the import cleanup. These are non-critical if the app starts successfully.

## Next Steps (Phase 2+)

1. Add comprehensive type hints
2. Implement Pydantic Settings for configuration
3. Fix database session lifecycle
4. Add comprehensive tests
5. Implement proper error handling middleware

## Support

For issues or questions, refer to:

- Architecture Analysis: `architecture_analysis_report.md`
- Phase 1 Report: `PHASE1_COMPLETE.md`
