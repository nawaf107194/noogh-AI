#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioral Monitoring & Anomaly Detection
==========================================

Pattern analysis and anomaly detection for system behavior monitoring.

Addresses Q87-Q88 from Functional Maturity Audit.

Features:
- Behavioral pattern tracking
- Statistical anomaly detection
- Baseline establishment
- Alert generation
- Time-series analysis
- Multi-dimensional monitoring
- Adaptive thresholds

Author: Noogh AI Team
Date: 2025-11-09
Priority: MEDIUM
"""

import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import json


# ============================================================================
# Enums
# ============================================================================

class AnomalyType(str, Enum):
    """Types of anomalies"""
    STATISTICAL_OUTLIER = "statistical_outlier"
    SUDDEN_SPIKE = "sudden_spike"
    SUDDEN_DROP = "sudden_drop"
    PATTERN_DEVIATION = "pattern_deviation"
    THRESHOLD_VIOLATION = "threshold_violation"
    UNUSUAL_FREQUENCY = "unusual_frequency"


class SeverityLevel(str, Enum):
    """Anomaly severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DataPoint:
    """Single data point for monitoring"""
    timestamp: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    metric_name: str
    observed_value: float
    expected_range: Tuple[float, float]
    deviation_score: float
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BehavioralBaseline:
    """Behavioral baseline statistics"""
    metric_name: str
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    sample_count: int
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# Behavioral Monitor
# ============================================================================

class BehavioralMonitor:
    """
    Behavioral monitoring and anomaly detection system
    """

    def __init__(
        self,
        window_size: int = 100,
        z_score_threshold: float = 3.0,
        baseline_min_samples: int = 20
    ):
        """
        Initialize behavioral monitor

        Args:
            window_size: Number of recent data points to keep
            z_score_threshold: Z-score threshold for outlier detection
            baseline_min_samples: Minimum samples to establish baseline
        """
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.baseline_min_samples = baseline_min_samples

        # Data storage (metric_name -> deque of values)
        self._data_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Baselines
        self._baselines: Dict[str, BehavioralBaseline] = {}

        # Detected anomalies
        self._anomalies: List[Anomaly] = []

        # Statistics
        self._data_points_received = 0
        self._anomalies_detected = 0

    def record_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """
        Record a metric value

        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        self._data_points_received += 1

        # Store data point
        data_point = DataPoint(
            timestamp=datetime.now(timezone.utc).isoformat(),
            metric_name=metric_name,
            value=value,
            metadata=metadata or {}
        )

        self._data_windows[metric_name].append(value)

        # Update baseline
        self._update_baseline(metric_name)

        # Check for anomalies
        anomalies = self._detect_anomalies(metric_name, value)
        self._anomalies.extend(anomalies)
        self._anomalies_detected += len(anomalies)

        return anomalies

    def _update_baseline(self, metric_name: str):
        """Update baseline statistics for a metric"""
        values = list(self._data_windows[metric_name])

        if len(values) < self.baseline_min_samples:
            return  # Not enough data

        baseline = BehavioralBaseline(
            metric_name=metric_name,
            mean=statistics.mean(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            sample_count=len(values)
        )

        self._baselines[metric_name] = baseline

    def _detect_anomalies(self, metric_name: str, value: float) -> List[Anomaly]:
        """
        Detect anomalies for a metric value

        Args:
            metric_name: Metric name
            value: Current value

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Get baseline
        baseline = self._baselines.get(metric_name)
        if not baseline:
            return anomalies  # No baseline yet

        # Z-score anomaly detection
        if baseline.std_dev > 0:
            z_score = abs((value - baseline.mean) / baseline.std_dev)

            if z_score > self.z_score_threshold:
                severity = SeverityLevel.CRITICAL if z_score > self.z_score_threshold * 1.5 else SeverityLevel.WARNING

                anomaly = Anomaly(
                    anomaly_id=f"A{self._anomalies_detected + 1:06d}",
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    metric_name=metric_name,
                    observed_value=value,
                    expected_range=(baseline.mean - 2 * baseline.std_dev, baseline.mean + 2 * baseline.std_dev),
                    deviation_score=z_score,
                    message=f"Statistical outlier: {metric_name} = {value:.2f} (z-score: {z_score:.2f})"
                )
                anomalies.append(anomaly)

        # Sudden spike detection (compared to recent average)
        values = list(self._data_windows[metric_name])
        if len(values) >= 10:
            recent_avg = statistics.mean(values[-10:-1])  # Last 9 values
            spike_ratio = value / recent_avg if recent_avg > 0 else 0

            if spike_ratio > 2.0:  # 2x increase
                anomaly = Anomaly(
                    anomaly_id=f"A{self._anomalies_detected + len(anomalies) + 1:06d}",
                    anomaly_type=AnomalyType.SUDDEN_SPIKE,
                    severity=SeverityLevel.WARNING,
                    metric_name=metric_name,
                    observed_value=value,
                    expected_range=(recent_avg * 0.5, recent_avg * 1.5),
                    deviation_score=spike_ratio,
                    message=f"Sudden spike: {metric_name} increased by {spike_ratio:.1f}x"
                )
                anomalies.append(anomaly)

            elif spike_ratio < 0.5 and recent_avg > 0:  # 50% decrease
                anomaly = Anomaly(
                    anomaly_id=f"A{self._anomalies_detected + len(anomalies) + 1:06d}",
                    anomaly_type=AnomalyType.SUDDEN_DROP,
                    severity=SeverityLevel.WARNING,
                    metric_name=metric_name,
                    observed_value=value,
                    expected_range=(recent_avg * 0.5, recent_avg * 1.5),
                    deviation_score=1.0 - spike_ratio,
                    message=f"Sudden drop: {metric_name} decreased by {(1 - spike_ratio) * 100:.1f}%"
                )
                anomalies.append(anomaly)

        return anomalies

    def get_baseline(self, metric_name: str) -> Optional[BehavioralBaseline]:
        """Get baseline for a metric"""
        return self._baselines.get(metric_name)

    def get_recent_anomalies(self, limit: int = 10, severity: Optional[SeverityLevel] = None) -> List[Anomaly]:
        """
        Get recent anomalies

        Args:
            limit: Maximum number of anomalies to return
            severity: Filter by severity level

        Returns:
            List of anomalies
        """
        anomalies = self._anomalies

        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]

        return anomalies[-limit:]

    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a metric

        Args:
            metric_name: Metric name

        Returns:
            Summary dictionary
        """
        baseline = self._baselines.get(metric_name)
        values = list(self._data_windows[metric_name])

        if not values:
            return {"error": "No data for metric"}

        return {
            "metric_name": metric_name,
            "current_value": values[-1] if values else None,
            "sample_count": len(values),
            "mean": baseline.mean if baseline else None,
            "std_dev": baseline.std_dev if baseline else None,
            "min": baseline.min_value if baseline else None,
            "max": baseline.max_value if baseline else None,
            "recent_trend": self._calculate_trend(values),
            "baseline_established": baseline is not None
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 5:
            return "insufficient_data"

        recent = statistics.mean(values[-5:])
        older = statistics.mean(values[-10:-5]) if len(values) >= 10 else statistics.mean(values[:-5])

        if recent > older * 1.1:
            return "increasing"
        elif recent < older * 0.9:
            return "decreasing"
        else:
            return "stable"

    def clear_anomalies(self):
        """Clear anomaly history"""
        self._anomalies.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "data_points_received": self._data_points_received,
            "anomalies_detected": self._anomalies_detected,
            "monitored_metrics": len(self._baselines),
            "anomaly_rate": round(self._anomalies_detected / self._data_points_received * 100, 2) if self._data_points_received > 0 else 0,
            "window_size": self.window_size,
            "z_score_threshold": self.z_score_threshold
        }


# ============================================================================
# Demo / Test
# ============================================================================

if __name__ == "__main__":
    print("Behavioral Monitoring & Anomaly Detection")
    print("=" * 70)

    # Create monitor
    monitor = BehavioralMonitor(
        window_size=50,
        z_score_threshold=2.5,
        baseline_min_samples=10
    )

    # Simulate normal behavior
    print("\n✅ Recording normal behavior (baseline establishment)...")
    import random
    random.seed(42)

    for i in range(30):
        # Normal CPU usage around 50%
        cpu_usage = 50 + random.gauss(0, 5)
        monitor.record_metric("cpu_usage", cpu_usage)

        # Normal request rate around 100 req/s
        req_rate = 100 + random.gauss(0, 10)
        monitor.record_metric("request_rate", req_rate)

    print(f"  Recorded 30 normal data points")

    # Check baseline
    cpu_baseline = monitor.get_baseline("cpu_usage")
    if cpu_baseline:
        print(f"  CPU baseline: mean={cpu_baseline.mean:.2f}, std_dev={cpu_baseline.std_dev:.2f}")

    # Simulate anomaly - CPU spike
    print("\n✅ Simulating CPU spike anomaly...")
    anomalies = monitor.record_metric("cpu_usage", 95.0)  # Sudden spike to 95%
    if anomalies:
        for anomaly in anomalies:
            print(f"  🚨 {anomaly.severity.value.upper()}: {anomaly.message}")

    # Simulate anomaly - Request rate drop
    print("\n✅ Simulating request rate drop...")
    anomalies = monitor.record_metric("request_rate", 20.0)  # Drop to 20 req/s
    if anomalies:
        for anomaly in anomalies:
            print(f"  🚨 {anomaly.severity.value.upper()}: {anomaly.message}")

    # Continue with more normal data
    print("\n✅ Recording more normal data...")
    for i in range(10):
        cpu_usage = 50 + random.gauss(0, 5)
        monitor.record_metric("cpu_usage", cpu_usage)

    # Get metric summary
    print("\n📊 Metric Summary (CPU Usage):")
    summary = monitor.get_metric_summary("cpu_usage")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Recent anomalies
    print("\n🚨 Recent Anomalies:")
    anomalies = monitor.get_recent_anomalies(limit=10)
    for anomaly in anomalies:
        print(f"  [{anomaly.anomaly_id}] {anomaly.anomaly_type.value}: {anomaly.message}")

    # Statistics
    print("\n" + "=" * 70)
    print("📊 Monitoring Statistics:")
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ Behavioral Monitoring system test completed!")
    print("=" * 70)
