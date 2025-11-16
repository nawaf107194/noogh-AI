"""
Phase 5: Health Baseline & Learning System

Learns what is "normal" for the system and detects deviations.
Uses SQLite for efficient time-series analysis.
"""

import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import statistics

from .system_monitor import SystemHealth

DB_PATH = Path("/home/noogh/noogh_unified_system/data/performance.db")


@dataclass
class BaselineMetrics:
    """Statistical baseline for a metric"""
    metric_name: str
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    sample_count: int
    last_updated: float

    def is_normal(self, value: float, threshold: float = 2.0) -> bool:
        """Check if value is within threshold * std_dev from mean"""
        if self.std_dev == 0:
            return abs(value - self.mean) < 0.01
        z_score = abs(value - self.mean) / self.std_dev
        return z_score < threshold

    def get_deviation(self, value: float) -> float:
        """Get number of standard deviations from mean"""
        if self.std_dev == 0:
            return 0.0
        return (value - self.mean) / self.std_dev


class HealthBaseline:
    """Learns and maintains baseline health metrics"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Metrics history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL
            )
        """)

        # Create index separately
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metric_time
            ON metric_history (metric_name, timestamp)
        """)

        # Baselines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                metric_name TEXT PRIMARY KEY,
                mean REAL NOT NULL,
                std_dev REAL NOT NULL,
                min_val REAL NOT NULL,
                max_val REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated REAL NOT NULL
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                alert_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                baseline_mean REAL,
                deviation REAL,
                message TEXT,
                acknowledged BOOLEAN DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def record_snapshot(self, health: SystemHealth):
        """Record a health snapshot to history"""
        metrics = {
            'cpu_percent': health.cpu_percent,
            'ram_percent': health.ram_percent,
            'disk_percent': health.disk_percent,
        }

        if health.gpu_available:
            metrics['gpu_temp'] = health.gpu_temp
            metrics['gpu_utilization'] = health.gpu_utilization
            metrics['vram_percent'] = health.vram_percent

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = time.time()
        for metric_name, value in metrics.items():
            if value is not None:
                cursor.execute(
                    "INSERT INTO metric_history (timestamp, metric_name, value) VALUES (?, ?, ?)",
                    (timestamp, metric_name, value)
                )

        conn.commit()
        conn.close()

    def update_baselines(self, lookback_hours: int = 24):
        """Update baseline statistics from recent history"""
        cutoff_time = time.time() - (lookback_hours * 3600)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all unique metrics
        cursor.execute("SELECT DISTINCT metric_name FROM metric_history WHERE timestamp > ?", (cutoff_time,))
        metrics = [row[0] for row in cursor.fetchall()]

        for metric_name in metrics:
            # Get values from lookback period
            cursor.execute(
                "SELECT value FROM metric_history WHERE metric_name = ? AND timestamp > ?",
                (metric_name, cutoff_time)
            )
            values = [row[0] for row in cursor.fetchall()]

            if len(values) < 10:  # Need at least 10 samples
                continue

            # Calculate statistics
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            min_val = min(values)
            max_val = max(values)

            # Update or insert baseline
            cursor.execute("""
                INSERT OR REPLACE INTO baselines
                (metric_name, mean, std_dev, min_val, max_val, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (metric_name, mean, std_dev, min_val, max_val, len(values), time.time()))

        conn.commit()
        conn.close()

    def get_baseline(self, metric_name: str) -> Optional[BaselineMetrics]:
        """Get baseline for a specific metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM baselines WHERE metric_name = ?", (metric_name,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return BaselineMetrics(
            metric_name=row[0],
            mean=row[1],
            std_dev=row[2],
            min_val=row[3],
            max_val=row[4],
            sample_count=row[5],
            last_updated=row[6]
        )

    def get_all_baselines(self) -> Dict[str, BaselineMetrics]:
        """Get all baselines as a dictionary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM baselines")
        rows = cursor.fetchall()
        conn.close()

        baselines = {}
        for row in rows:
            baseline = BaselineMetrics(
                metric_name=row[0],
                mean=row[1],
                std_dev=row[2],
                min_val=row[3],
                max_val=row[4],
                sample_count=row[5],
                last_updated=row[6]
            )
            baselines[row[0]] = baseline

        return baselines

    def check_deviations(self, health: SystemHealth, threshold: float = 2.0) -> List[Dict]:
        """Check current health against baselines"""
        deviations = []

        metrics = {
            'cpu_percent': health.cpu_percent,
            'ram_percent': health.ram_percent,
            'disk_percent': health.disk_percent,
        }

        if health.gpu_available:
            metrics['gpu_temp'] = health.gpu_temp
            metrics['gpu_utilization'] = health.gpu_utilization
            metrics['vram_percent'] = health.vram_percent

        for metric_name, value in metrics.items():
            if value is None:
                continue

            baseline = self.get_baseline(metric_name)
            if not baseline:
                continue

            if not baseline.is_normal(value, threshold):
                deviation = baseline.get_deviation(value)
                deviations.append({
                    'metric': metric_name,
                    'current': value,
                    'baseline_mean': baseline.mean,
                    'baseline_std': baseline.std_dev,
                    'deviation': deviation,
                    'severity': 'HIGH' if abs(deviation) > 3 else 'MEDIUM'
                })

        return deviations

    def record_alert(self, alert_type: str, metric_name: str, value: float,
                     baseline_mean: Optional[float], deviation: Optional[float], message: str):
        """Record an alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alerts (timestamp, alert_type, metric_name, value, baseline_mean, deviation, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (time.time(), alert_type, metric_name, value, baseline_mean, deviation, message))

        conn.commit()
        conn.close()

    def get_recent_alerts(self, hours: int = 24, acknowledged: bool = False) -> List[Dict]:
        """Get recent alerts"""
        cutoff_time = time.time() - (hours * 3600)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, timestamp, alert_type, metric_name, value, baseline_mean, deviation, message, acknowledged
            FROM alerts
            WHERE timestamp > ? AND acknowledged = ?
            ORDER BY timestamp DESC
        """, (cutoff_time, 1 if acknowledged else 0))

        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'id': row[0],
                'timestamp': row[1],
                'alert_type': row[2],
                'metric': row[3],
                'value': row[4],
                'baseline_mean': row[5],
                'deviation': row[6],
                'message': row[7],
                'acknowledged': bool(row[8])
            })

        conn.close()
        return alerts

    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Tuple[float, float]]:
        """Get time-series data for a metric"""
        cutoff_time = time.time() - (hours * 3600)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, value FROM metric_history
            WHERE metric_name = ? AND timestamp > ?
            ORDER BY timestamp ASC
        """, (metric_name, cutoff_time))

        history = cursor.fetchall()
        conn.close()

        return history

    def cleanup_old_data(self, keep_days: int = 7):
        """Remove old metric history to save space"""
        cutoff_time = time.time() - (keep_days * 24 * 3600)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM metric_history WHERE timestamp < ?", (cutoff_time,))
        deleted = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted


# Testing
if __name__ == "__main__":
    from system_monitor import SystemMonitor

    print("=" * 70)
    print("🧠 Testing Health Baseline System")
    print("=" * 70)

    baseline = HealthBaseline()
    monitor = SystemMonitor()

    # Record current snapshot
    health = monitor.capture_snapshot()
    baseline.record_snapshot(health)
    print("\n✅ Recorded snapshot to database")

    # Update baselines (won't have much data yet)
    baseline.update_baselines(lookback_hours=1)
    print("✅ Updated baselines")

    # Get baselines
    baselines = baseline.get_all_baselines()
    print(f"\n📊 Current Baselines: {len(baselines)} metrics")
    for name, bl in baselines.items():
        print(f"   {name}: μ={bl.mean:.1f}, σ={bl.std_dev:.1f} (n={bl.sample_count})")

    # Check for deviations
    deviations = baseline.check_deviations(health, threshold=2.0)
    print(f"\n⚠️  Deviations: {len(deviations)}")
    for dev in deviations:
        print(f"   {dev['metric']}: {dev['current']:.1f} (baseline: {dev['baseline_mean']:.1f}, σ={dev['deviation']:.2f})")

    print("\n" + "=" * 70)
