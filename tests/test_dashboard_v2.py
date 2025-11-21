import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock Streamlit and other UI libs BEFORE importing dashboard
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit_autorefresh'] = MagicMock()
sys.modules['streamlit_mic_recorder'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['pandas'] = MagicMock()

# Configure st.columns to return a list of mocks
def mock_columns(spec):
    count = spec if isinstance(spec, int) else len(spec)
    return [MagicMock() for _ in range(count)]

sys.modules['streamlit'].columns.side_effect = mock_columns
sys.modules['streamlit'].tabs.side_effect = mock_columns # Tabs behaves similarly
sys.modules['streamlit'].chat_input.return_value = None # Prevent execution loop
sys.modules['streamlit_mic_recorder'].mic_recorder.return_value = None # Prevent execution loop

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dashboard module
import src.interface.dashboard as dashboard

# Configure Mock Pandas DataFrame
class MockDataFrame:
    def __init__(self, data=None, **kwargs):
        self.data = data if data else []
        self.empty = False if data else True
        self.columns = kwargs.get('columns', [])
    
    def __getitem__(self, item):
        return self
        
    def __setitem__(self, key, value):
        pass
        
    def sort_values(self, *args, **kwargs):
        return self
        
    def __len__(self):
        return len(self.data)

dashboard.pd.DataFrame = MockDataFrame
dashboard.pd.to_datetime = MagicMock(return_value="2025-01-01")

class TestDashboardV2(unittest.TestCase):
    
    def setUp(self):
        # Setup temp paths
        self.test_dir = Path("/tmp/noogh_test")
        # Clean up if exists
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # Mock settings
        dashboard.settings = MagicMock()
        dashboard.settings.data_dir = self.test_dir
        dashboard.project_root = self.test_dir
        
        # Create config dir
        (self.test_dir / "config").mkdir(exist_ok=True)

    def test_minister_state_persistence(self):
        print("\n🔹 Testing Minister State Persistence...")
        
        # Test Save
        state = {"finance": False, "health": True}
        dashboard.save_minister_state(state)
        
        config_path = self.test_dir / "config" / "minister_state.json"
        self.assertTrue(config_path.exists())
        
        with open(config_path, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data, state)
        print("✅ Save State: PASSED")
        
        # Test Load
        loaded_state = dashboard.load_minister_state()
        self.assertEqual(loaded_state, state)
        print("✅ Load State: PASSED")
        
        # Test Default Load (Missing File)
        config_path.unlink()
        default_state = dashboard.load_minister_state()
        self.assertTrue(default_state['finance']) # Default is True
        print("✅ Default State: PASSED")

    def test_trading_performance_logic(self):
        print("\n🔹 Testing Trading Performance Logic...")
        
        # 1. Test Missing Ledger
        metrics, df = dashboard.get_trading_performance()
        self.assertIsNone(metrics)
        self.assertIsNone(df)
        print("✅ Missing Ledger: PASSED (Handled gracefully)")
        
        # 2. Test Valid Ledger
        ledger_data = {
            "balances": {"spot_usdt": 5000, "futures_usdt": 5000},
            "stats": {"total_trades": 10, "profitable_trades": 6, "total_pnl": 500.0},
            "trades": [
                {"timestamp": "2025-01-01T12:00:00", "status": "OPEN", "market_type": "SPOT"},
                {"timestamp": "2025-01-02T12:00:00", "status": "CLOSED", "market_type": "FUTURES"}
            ]
        }
        
        ledger_path = self.test_dir / "paper_ledger.json"
        with open(ledger_path, 'w') as f:
            json.dump(ledger_data, f)
            
        metrics, df = dashboard.get_trading_performance()
        
        self.assertEqual(metrics['total_value'], 10000)
        self.assertEqual(metrics['win_rate'], 60.0)
        self.assertEqual(metrics['active_trades'], 1)
        self.assertEqual(metrics['total_pnl'], 500.0)
        self.assertEqual(len(df), 2)
        print("✅ Valid Ledger: PASSED")

if __name__ == '__main__':
    unittest.main()
