#!/usr/bin/env python3
"""
🗄️ Pattern Database
قاعدة بيانات الأنماط

Store and retrieve detected patterns for analysis
"""

import sqlite3
import json
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import logging
import numpy as np

from .pattern_detector import DetectedPattern, PatternType

logger = logging.getLogger(__name__)


class PatternDatabase:
    """
    SQLite database for storing detected patterns

    Stores:
    - Pattern detections
    - Performance tracking
    - Historical patterns
    """

    def __init__(self, db_path: str = 'data/trading/patterns.db'):
        """
        Initialize pattern database

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self._convert_to_json_serializable(item) for item in obj.tolist()]
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            # For any other type, try to convert to string
            try:
                return str(obj)
            except Exception as e:
                return None

    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    start_idx INTEGER NOT NULL,
                    end_idx INTEGER NOT NULL,
                    key_points TEXT,
                    target_price REAL,
                    stop_loss REAL,
                    strength REAL NOT NULL,
                    description TEXT,
                    timeframe TEXT,
                    candles_analyzed INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    result TEXT,
                    pnl_percentage REAL,
                    exit_timestamp TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_symbol
                ON patterns(symbol)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_timestamp
                ON patterns(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_type
                ON patterns(pattern_type)
            """)

            conn.commit()

        logger.info(f"✅ Pattern database initialized: {self.db_path}")

    def store_pattern(
        self,
        pattern: DetectedPattern,
        symbol: str,
        timeframe: str = '1d',
        candles_analyzed: int = 0
    ) -> int:
        """
        Store detected pattern

        Args:
            pattern: Detected pattern
            symbol: Trading symbol
            timeframe: Timeframe analyzed
            candles_analyzed: Number of candles

        Returns:
            Pattern ID
        """
        with sqlite3.connect(self.db_path) as conn:
            # Convert key_points to JSON-serializable format
            key_points_serializable = self._convert_to_json_serializable(pattern.key_points)

            cursor = conn.execute("""
                INSERT INTO patterns (
                    timestamp, symbol, pattern_type, direction,
                    confidence, start_idx, end_idx, key_points,
                    target_price, stop_loss, strength, description,
                    timeframe, candles_analyzed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                symbol,
                pattern.pattern_type.value,
                pattern.direction,
                pattern.confidence,
                int(pattern.start_idx) if pattern.start_idx is not None else None,
                int(pattern.end_idx) if pattern.end_idx is not None else None,
                json.dumps(key_points_serializable),
                float(pattern.target_price) if pattern.target_price is not None else None,
                float(pattern.stop_loss) if pattern.stop_loss is not None else None,
                pattern.strength,
                pattern.description,
                timeframe,
                candles_analyzed
            ))

            conn.commit()
            pattern_id = cursor.lastrowid

        logger.info(f"   💾 Pattern stored: ID={pattern_id}, {pattern.pattern_type.value}")

        return pattern_id

    def get_patterns_by_symbol(
        self,
        symbol: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get patterns for a symbol

        Args:
            symbol: Trading symbol
            limit: Maximum number of patterns

        Returns:
            List of pattern dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT * FROM patterns
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))

            patterns = [dict(row) for row in cursor.fetchall()]

        return patterns

    def get_pattern_statistics(
        self,
        pattern_type: Optional[str] = None
    ) -> Dict:
        """
        Get pattern detection statistics

        Args:
            pattern_type: Optional pattern type filter

        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            if pattern_type:
                query = """
                    SELECT
                        COUNT(*) as total,
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence,
                        AVG(strength) as avg_strength
                    FROM patterns
                    WHERE pattern_type = ?
                """
                result = conn.execute(query, (pattern_type,)).fetchone()
            else:
                query = """
                    SELECT
                        COUNT(*) as total,
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence,
                        AVG(strength) as avg_strength
                    FROM patterns
                """
                result = conn.execute(query).fetchone()

            stats = {
                'total_patterns': result[0],
                'avg_confidence': result[1] or 0.0,
                'min_confidence': result[2] or 0.0,
                'max_confidence': result[3] or 0.0,
                'avg_strength': result[4] or 0.0
            }

        return stats

    def get_pattern_distribution(self) -> Dict:
        """
        Get distribution of pattern types

        Returns:
            Dictionary mapping pattern type to count
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT pattern_type, COUNT(*) as count
                FROM patterns
                GROUP BY pattern_type
                ORDER BY count DESC
            """)

            distribution = {row[0]: row[1] for row in cursor.fetchall()}

        return distribution

    def record_pattern_performance(
        self,
        pattern_id: int,
        entry_price: float,
        exit_price: Optional[float] = None,
        result: Optional[str] = None
    ):
        """
        Record pattern performance

        Args:
            pattern_id: Pattern ID
            entry_price: Entry price
            exit_price: Exit price (if closed)
            result: 'WIN' or 'LOSS' (if closed)
        """
        pnl = None
        if exit_price:
            pnl = ((exit_price - entry_price) / entry_price) * 100

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pattern_performance (
                    pattern_id, entry_price, exit_price, result,
                    pnl_percentage, exit_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pattern_id,
                entry_price,
                exit_price,
                result,
                pnl,
                datetime.now().isoformat() if exit_price else None
            ))

            conn.commit()

        logger.info(f"   📊 Performance recorded for pattern {pattern_id}")


# TODO: Add more database features
# - Pattern correlation analysis
# - Success rate tracking
# - Backtesting results storage
# - Pattern similarity search
