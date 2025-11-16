#!/usr/bin/env python3
"""
🤖 Autonomous System Client
واجهة تواصل مباشرة بين الدماغ والنظام الذاتي

Features:
    ✅ Automatic retry with exponential backoff
    ✅ Configurable timeout and retries
    ✅ Smart error handling (4xx vs 5xx)
    ✅ Detailed logging for debugging
    ✅ Production-grade reliability

Usage:
from api.utils.autonomous_client import AutonomousClient

    # With retry configuration
    client = AutonomousClient(
        timeout=30.0,       # 30 second timeout per request
        max_retries=3,      # Retry up to 3 times
        retry_backoff=1.0,  # 1s → 2s → 4s exponential backoff
        verbose=True        # Enable logging
    )

    # Check resources before training
    resources = await client.get_resources()
    if resources['gpu_memory_percent'] < 80:
        await client.prepare_training("MyModel")
        await client.start_training(epochs=10)
"""

import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AutonomousClient:
    """
    🤖 عميل للتواصل مع النظام الذاتي

    يتيح للدماغ:
    - التحقق من الموارد قبل التدريب
    - طلب قرارات من موزع الحمل
    - بدء/إيقاف جلسات التدريب
    - مراقبة حالة النظام

    Args:
        base_url: API base URL (default: http://localhost:8000/api/autonomous)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_backoff: Initial backoff time for exponential backoff (default: 1.0)
                      Actual wait times: 1s → 2s → 4s for 3 retries
        verbose: Enable detailed logging (default: False)

    Features:
        ✅ Automatic retry with exponential backoff
        ✅ Smart error handling (skip retry on 4xx client errors)
        ✅ Timeout protection on each request
        ✅ Detailed logging for production debugging

    Example:
        async with AutonomousClient(max_retries=5, verbose=True) as client:
            resources = await client.get_resources()
            if resources['gpu_memory_percent'] < 80:
                await client.prepare_training("MyModel", estimated_vram=4.0)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/api/autonomous",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        verbose: bool = False
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.verbose = verbose
        self._session: Optional[aiohttp.ClientSession] = None

        if self.verbose:
            logger.info(f"🤖 AutonomousClient initialized: {self.base_url}")
            logger.info(f"   Max retries: {max_retries}, Backoff: {retry_backoff}s, Timeout: {timeout}s")

    async def __aenter__(self):
        """Context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and exponential backoff

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., '/resources')
            data: JSON data for POST requests
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            aiohttp.ClientError: After all retries exhausted
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                if self.verbose and attempt > 0:
                    logger.info(f"🔄 Retry {attempt}/{self.max_retries - 1}: {method} {endpoint}")

                if method.upper() == "GET":
                    async with self._session.get(url, params=params) as response:
                        response.raise_for_status()
                        result = await response.json()

                        if self.verbose and attempt > 0:
                            logger.info(f"✅ Request succeeded on retry {attempt}")

                        return result

                elif method.upper() == "POST":
                    async with self._session.post(url, json=data, params=params) as response:
                        response.raise_for_status()
                        result = await response.json()

                        if self.verbose and attempt > 0:
                            logger.info(f"✅ Request succeeded on retry {attempt}")

                        return result

                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

            except asyncio.TimeoutError as e:
                last_exception = e
                wait_time = self.retry_backoff * (2 ** attempt)  # Exponential backoff

                if attempt < self.max_retries - 1:
                    logger.warning(f"⏱️  Timeout on {method} {endpoint}, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Request timed out after {self.max_retries} attempts: {method} {url}")

            except aiohttp.ClientError as e:
                last_exception = e

                # Don't retry on 4xx errors (client errors)
                if hasattr(e, 'status') and 400 <= e.status < 500:
                    logger.error(f"❌ Client error {e.status}: {method} {url}")
                    raise

                # Retry on 5xx errors (server errors) and network errors
                wait_time = self.retry_backoff * (2 ** attempt)

                if attempt < self.max_retries - 1:
                    logger.warning(f"⚠️  Request failed: {e}, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Request failed after {self.max_retries} attempts: {method} {url} - {e}")

            except Exception as e:
                last_exception = e
                logger.error(f"❌ Unexpected error: {e}")
                raise

        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Request failed after {self.max_retries} attempts")

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to autonomous API (with retry logic)

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., '/resources')
            data: JSON data for POST requests
            params: Query parameters

        Returns:
            Response JSON
        """
        return await self._request_with_retry(method, endpoint, data, params)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Status & Monitoring
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_resources(self) -> Dict[str, Any]:
        """
        📊 الحصول على حالة الموارد الحالية
        Get current resources (VRAM, CPU, RAM, Temperature)

        Returns:
            {
                'gpu_memory_percent': 66.7,
                'gpu_temperature': 34.0,
                'cpu_percent': 1.5,
                'ram_percent': 22.7,
                'overall_status': 'normal'
            }
        """
        response = await self._request("GET", "/resources")
        return response.get("resources", {})

    async def get_status(self) -> Dict[str, Any]:
        """
        🤖 الحصول على الحالة الكاملة
        Get full autonomous system status

        Returns:
            Complete system status including manager, monitoring, etc.
        """
        response = await self._request("GET", "/status")
        return response.get("status", {})

    async def get_statistics(self) -> Dict[str, Any]:
        """
        📈 الحصول على الإحصائيات
        Get complete statistics

        Returns:
            Statistics for manager, resource monitor, load balancer
        """
        response = await self._request("GET", "/statistics")
        return response.get("statistics", {})

    async def get_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        📜 الحصول على تاريخ القرارات
        Get decision history

        Args:
            limit: Number of decisions to return

        Returns:
            List of recent decisions
        """
        response = await self._request("GET", "/decisions", params={"limit": limit})
        return response.get("decisions", [])

    async def health_check(self) -> bool:
        """
        ❤️  فحص صحة النظام
        Check if autonomous system is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self._request("GET", "/health")
            return response.get("success", False)
        except Exception:
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Control
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def start_monitoring(self) -> bool:
        """
        ▶️  بدء المراقبة الذاتية
        Start autonomous monitoring

        Returns:
            True if successful
        """
        try:
            response = await self._request("POST", "/start")
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False

    async def stop_monitoring(self) -> bool:
        """
        ⏹️  إيقاف المراقبة الذاتية
        Stop autonomous monitoring

        Returns:
            True if successful
        """
        try:
            response = await self._request("POST", "/stop")
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Load Balancer
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def decide_device(
        self,
        task_type: str,
        estimated_vram: float = 0.5,
        priority: str = "medium",
        minister_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ⚖️  طلب قرار من موزع الحمل
        Request device allocation decision (CPU vs GPU)

        Args:
            task_type: Type of task (e.g., "training", "inference")
            estimated_vram: Estimated VRAM needed in GB
            priority: Priority level ("low", "medium", "high", "critical")
            minister_id: Optional minister ID

        Returns:
            {
                'device': 'gpu' or 'cpu',
                'reason': 'explanation',
                'task_id': 'task_xxx'
            }
        """
        data = {
            "task_type": task_type,
            "estimated_vram": estimated_vram,
            "priority": priority
        }

        if minister_id:
            data["minister_id"] = minister_id

        response = await self._request("POST", "/loadbalancer/decide", data=data)
        return response.get("decision", {})

    async def should_use_gpu(
        self,
        task_type: str,
        estimated_vram: float = 0.5,
        priority: str = "medium"
    ) -> bool:
        """
        🎯 هل يجب استخدام GPU؟
        Simple helper: Should this task use GPU?

        Args:
            task_type: Type of task
            estimated_vram: Estimated VRAM needed
            priority: Priority level

        Returns:
            True if GPU recommended, False if CPU
        """
        decision = await self.decide_device(task_type, estimated_vram, priority)
        return decision.get("device", "cpu").lower() == "gpu"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def prepare_training(
        self,
        model_name: str,
        estimated_vram_needed: float = 4.0,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        🔧 تحضير النظام للتدريب
        Prepare system for training

        This will:
        - Pause GPU ministers if needed
        - Unload ALLaM model
        - Clear PyTorch cache
        - Free VRAM

        Args:
            model_name: Name of model to train
            estimated_vram_needed: Estimated VRAM needed in GB
            force: Force preparation even if resources available

        Returns:
            {
                'success': True,
                'freed_vram': 2.5,
                'available_vram': 6.5,
                'paused_ministers': 3
            }
        """
        data = {
            "model_name": model_name,
            "estimated_vram_needed": estimated_vram_needed,
            "force": force
        }

        response = await self._request("POST", "/training/prepare", data=data)
        return response.get("preparation", {})

    async def start_training(self, epochs: int = 10) -> Dict[str, Any]:
        """
        🚀 بدء التدريب
        Start training session

        Note: Must call prepare_training() first

        Args:
            epochs: Number of epochs

        Returns:
            Training session info
        """
        data = {"epochs": epochs}
        response = await self._request("POST", "/training/start", data=data)
        return response.get("session", {})

    async def get_training_status(self) -> Optional[Dict[str, Any]]:
        """
        📊 حالة التدريب
        Get current training status

        Returns:
            Training status or None if no active session
        """
        response = await self._request("GET", "/training/status")
        return response.get("training")

    async def complete_training(self, success: bool = True) -> Dict[str, Any]:
        """
        🏁 إنهاء التدريب
        Complete training session

        This will:
        - Resume paused ministers
        - Reload ALLaM model
        - Restore system

        Args:
            success: Whether training was successful

        Returns:
            Training completion result
        """
        response = await self._request(
            "POST",
            "/training/complete",
            params={"success": success}
        )
        return response.get("result", {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # High-Level Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def is_ready_for_training(
        self,
        estimated_vram: float = 4.0,
        max_vram_threshold: float = 80.0
    ) -> bool:
        """
        ✅ هل النظام جاهز للتدريب؟
        Check if system is ready for training

        Args:
            estimated_vram: VRAM needed for training
            max_vram_threshold: Maximum acceptable VRAM usage %

        Returns:
            True if ready, False otherwise
        """
        resources = await self.get_resources()

        vram_percent = resources.get("gpu_memory_percent", 100)
        vram_total = resources.get("gpu_memory_total", 0)
        vram_used = resources.get("gpu_memory_used", 0)

        available_vram = vram_total - vram_used

        # Check if we have enough VRAM
        if available_vram < estimated_vram:
            return False

        # Check if VRAM usage is below threshold
        if vram_percent > max_vram_threshold:
            return False

        # Check overall status
        status = resources.get("overall_status", "unknown")
        if status not in ["normal", "warning"]:
            return False

        return True

    async def wait_for_resources(
        self,
        estimated_vram: float = 4.0,
        max_wait_seconds: float = 300.0,
        check_interval: float = 5.0
    ) -> bool:
        """
        ⏳ انتظار توفر الموارد
        Wait for resources to become available

        Args:
            estimated_vram: VRAM needed
            max_wait_seconds: Maximum time to wait
            check_interval: Check interval in seconds

        Returns:
            True if resources became available, False if timeout
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check if ready
            if await self.is_ready_for_training(estimated_vram):
                return True

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                return False

            # Wait before next check
            await asyncio.sleep(check_interval)

    async def auto_train(
        self,
        model_name: str,
        epochs: int = 10,
        estimated_vram: float = 4.0,
        wait_for_resources: bool = True,
        max_wait: float = 300.0
    ) -> Dict[str, Any]:
        """
        🤖 تدريب تلقائي كامل
        Fully automated training

        This will:
        1. Check if resources available
        2. Wait for resources if needed
        3. Prepare system
        4. Start training
        5. Monitor progress (caller's responsibility)
        6. Complete training (caller's responsibility)

        Args:
            model_name: Model to train
            epochs: Number of epochs
            estimated_vram: VRAM needed
            wait_for_resources: Wait if resources not available
            max_wait: Maximum wait time

        Returns:
            {
                'success': True,
                'session_id': 'training_xxx',
                'preparation': {...},
                'session': {...}
            }
        """
        result = {
            'success': False,
            'error': None,
            'preparation': None,
            'session': None
        }

        try:
            # Check resources
            if not await self.is_ready_for_training(estimated_vram):
                if wait_for_resources:
                    if self.verbose:
                        logger.info(f"⏳ Waiting for resources...")

                    if not await self.wait_for_resources(estimated_vram, max_wait):
                        result['error'] = "Timeout waiting for resources"
                        return result
                else:
                    result['error'] = "Resources not available"
                    return result

            # Prepare
            if self.verbose:
                logger.info(f"🔧 Preparing for training: {model_name}")

            prep = await self.prepare_training(model_name, estimated_vram)
            result['preparation'] = prep

            if not prep.get('success'):
                result['error'] = "Preparation failed"
                return result

            # Start
            if self.verbose:
                logger.info(f"🚀 Starting training: {epochs} epochs")

            session = await self.start_training(epochs)
            result['session'] = session
            result['success'] = True

            return result

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Auto-train failed: {e}")
            return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Self-Evaluation Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def log_training_session(
        self,
        model_name: str,
        device_used: str,
        duration_seconds: float,
        epochs: int,
        success: bool,
        resources_before: Dict[str, Any],
        resources_after: Dict[str, Any],
        estimated_vram_gb: float,
        lb_recommendation: str,
        lb_confidence: float,
        ministers_paused: bool = True,
        ministers_count: int = 0,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        📝 تسجيل جلسة تدريب
        Log a training session for evaluation

        Args:
            model_name: Model name
            device_used: "cpu" or "gpu"
            duration_seconds: Training duration
            epochs: Number of epochs
            success: Was training successful?
            resources_before: Resources snapshot before training
            resources_after: Resources snapshot after training
            estimated_vram_gb: Estimated VRAM needed
            lb_recommendation: Load balancer recommendation ("cpu" or "gpu")
            lb_confidence: Load balancer confidence (0-1)
            ministers_paused: Were ministers paused?
            ministers_count: How many ministers were paused
            error_message: Error message if failed

        Returns:
            {
                'success': True,
                'session_id': 'abc123',
                'message': '...'
            }
        """
        data = {
            "model_name": model_name,
            "device_used": device_used,
            "duration_seconds": duration_seconds,
            "epochs": epochs,
            "success": success,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "estimated_vram_gb": estimated_vram_gb,
            "lb_recommendation": lb_recommendation,
            "lb_confidence": lb_confidence,
            "ministers_paused": ministers_paused,
            "ministers_count": ministers_count,
            "error_message": error_message
        }

        response = await self._request("POST", "/evaluation/log", data=data)
        return response

    async def get_evaluation_analysis(self, days: int = 7) -> Dict[str, Any]:
        """
        📊 تحليل الأداء
        Get performance analysis

        Args:
            days: Number of days to analyze

        Returns:
            Performance analysis
        """
        response = await self._request("GET", "/evaluation/analysis", params={'days': days})
        return response.get("analysis", {})

    async def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        💡 الحصول على التوصيات
        Get recommendations for improving system performance

        Returns:
            List of recommendations
        """
        response = await self._request("GET", "/evaluation/recommendations")
        return response.get("recommendations", [])

    async def get_insights(self) -> Dict[str, Any]:
        """
        🧠 الرؤى الذكية
        Get high-level insights about system health

        Returns:
            Insights dictionary
        """
        response = await self._request("GET", "/evaluation/insights")
        return response.get("insights", {})

    async def get_weekly_summary(self) -> Dict[str, Any]:
        """
        📅 الملخص الأسبوعي
        Generate weekly summary report

        Returns:
            Weekly summary
        """
        response = await self._request("GET", "/evaluation/weekly-summary")
        return response.get("summary", {})

    async def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        📈 إحصائيات التقييم
        Get evaluation system statistics

        Returns:
            Statistics about logged sessions
        """
        response = await self._request("GET", "/evaluation/statistics")
        return response.get("statistics", {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ML Prediction Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def train_ml_predictor(self) -> Dict[str, Any]:
        """
        🎓 تدريب محرك التنبؤ
        Train ML predictor on historical data

        This trains the ML model on all logged training sessions.

        Returns:
            Training results with accuracy metrics
        """
        response = await self._request("POST", "/ml/train")
        return response

    async def predict_vram(
        self,
        model_name: str,
        epochs: int = 10,
        device: str = "gpu",
        ministers_count: int = 0,
        safety_margin: float = 1.1
    ) -> Dict[str, Any]:
        """
        🔮 التنبؤ بـ VRAM المطلوب
        Predict VRAM required for training

        Uses ML model trained on historical data to predict VRAM needs.

        Args:
            model_name: Model name
            epochs: Number of epochs
            device: "cpu" or "gpu"
            ministers_count: Number of paused ministers
            safety_margin: Safety margin multiplier (default: 1.1 = +10%)

        Returns:
            {
                'predicted_vram_gb': float,
                'with_margin_gb': float,
                'confidence': float,
                'predicted_duration_seconds': float
            }
        """
        data = {
            "model_name": model_name,
            "epochs": epochs,
            "device": device,
            "ministers_count": ministers_count,
            "safety_margin": safety_margin
        }

        response = await self._request("POST", "/ml/predict-vram", data=data)
        return response.get("prediction", {})

    async def get_ml_model_info(self) -> Dict[str, Any]:
        """
        ℹ️  معلومات النموذج
        Get ML model information

        Returns:
            Model status, accuracy, training date, etc.
        """
        response = await self._request("GET", "/ml/model-info")
        return response.get("model_info", {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Auto-Tuning
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def autotune_for_model(
        self,
        model_name: str,
        epochs: int = 10,
        current_vram_gb: Optional[float] = None,
        current_batch_size: Optional[int] = 32,
        current_learning_rate: Optional[float] = 1e-4,
        current_safety_margin: Optional[float] = 1.1,
        current_device: Optional[str] = "gpu",
        current_pause_ministers: Optional[bool] = True
    ) -> Dict[str, Any]:
        """
        🎯 ضبط تلقائي للبارامترات
        Auto-tune training parameters for a model

        Uses ML predictions, evaluation feedback, and current resources
        to optimize VRAM allocation, batch size, learning rate, etc.

        Args:
            model_name: Name of model to train
            epochs: Number of training epochs
            current_vram_gb: Current VRAM allocation (None = auto)
            current_batch_size: Current batch size
            current_learning_rate: Current learning rate
            current_safety_margin: Current safety margin
            current_device: Current device (gpu/cpu)
            current_pause_ministers: Current ministers pause setting

        Returns:
            TuningProfile with optimized parameters and recommendations

        Example:
            profile = await client.autotune_for_model(
                model_name="BERT-Base",
                epochs=10,
                current_vram_gb=None  # Auto-tune VRAM
            )

            print(f"Recommended VRAM: {profile['vram_gb']:.2f} GB")
            print(f"Recommended batch size: {profile['batch_size']}")
            print(f"Success rate: {profile['estimated_success_rate']:.1%}")

            for rec in profile['recommendations']:
                print(f"  - {rec['parameter']}: {rec['recommended_value']} ({rec['reason']})")
        """
        payload = {
            "model_name": model_name,
            "epochs": epochs,
            "current_vram_gb": current_vram_gb,
            "current_batch_size": current_batch_size,
            "current_learning_rate": current_learning_rate,
            "current_safety_margin": current_safety_margin,
            "current_device": current_device,
            "current_pause_ministers": current_pause_ministers
        }
        response = await self._request("POST", "/autotune/tune", json=payload)
        return response.get("profile", {})

    async def get_autotune_history(
        self,
        model_name: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        📜 سجل الضبط التلقائي
        Get auto-tuning history

        Args:
            model_name: Optional model name to filter by
            days: Number of days to look back

        Returns:
            List of historical tuning profiles

        Example:
            history = await client.get_autotune_history(model_name="BERT-Base", days=7)

            for profile in history:
                print(f"{profile['timestamp']}: {profile['model_name']}")
                print(f"  VRAM: {profile['vram_gb']:.2f} GB")
                print(f"  Success rate: {profile['estimated_success_rate']:.1%}")
        """
        params = {"days": days}
        if model_name:
            params["model_name"] = model_name

        response = await self._request("GET", "/autotune/history", params=params)
        return response.get("profiles", [])

    async def get_autotuner_info(self) -> Dict[str, Any]:
        """
        ℹ️  معلومات نظام الضبط التلقائي
        Get auto-tuner system information

        Returns:
            Auto-tuner configuration and constraints

        Example:
            info = await client.get_autotuner_info()

            print(f"Storage: {info['storage_path']}")
            print(f"Constraints:")
            for key, value in info['constraints'].items():
                print(f"  {key}: {value}")
        """
        response = await self._request("GET", "/autotune/info")
        return response.get("info", {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Info
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_info(self) -> Dict[str, Any]:
        """Get system info"""
        response = await self._request("GET", "/info")
        return response.get("info", {})

    def __repr__(self):
        return f"AutonomousClient(base_url='{self.base_url}')"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Synchronous Wrapper (Optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SyncAutonomousClient:
    """
    Synchronous wrapper for AutonomousClient

    Args:
        base_url: API base URL (default: http://localhost:8000/api/autonomous)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum number of retries (default: 3)
        retry_backoff: Initial backoff time in seconds (default: 1.0)
        verbose: Enable verbose logging (default: False)

    Usage:
        client = SyncAutonomousClient(max_retries=5, verbose=True)
        resources = client.get_resources()
    """

    def __init__(self, **kwargs):
        self._async_client = AutonomousClient(**kwargs)
        self._loop = None

    def _get_loop(self):
        """Get or create event loop"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        """Run async coroutine synchronously"""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def get_resources(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_resources())

    def get_status(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_status())

    def decide_device(self, task_type: str, **kwargs) -> Dict[str, Any]:
        return self._run(self._async_client.decide_device(task_type, **kwargs))

    def prepare_training(self, model_name: str, **kwargs) -> Dict[str, Any]:
        return self._run(self._async_client.prepare_training(model_name, **kwargs))

    def start_training(self, epochs: int = 10) -> Dict[str, Any]:
        return self._run(self._async_client.start_training(epochs))

    def get_training_status(self) -> Optional[Dict[str, Any]]:
        return self._run(self._async_client.get_training_status())

    def complete_training(self, success: bool = True) -> Dict[str, Any]:
        return self._run(self._async_client.complete_training(success))

    # Evaluation methods
    def log_training_session(self, **kwargs) -> Dict[str, Any]:
        return self._run(self._async_client.log_training_session(**kwargs))

    def get_evaluation_analysis(self, days: int = 7) -> Dict[str, Any]:
        return self._run(self._async_client.get_evaluation_analysis(days))

    def get_recommendations(self) -> List[Dict[str, Any]]:
        return self._run(self._async_client.get_recommendations())

    def get_insights(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_insights())

    def get_weekly_summary(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_weekly_summary())

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_evaluation_statistics())

    # ML methods
    def train_ml_predictor(self) -> Dict[str, Any]:
        return self._run(self._async_client.train_ml_predictor())

    def predict_vram(self, **kwargs) -> Dict[str, Any]:
        return self._run(self._async_client.predict_vram(**kwargs))

    def get_ml_model_info(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_ml_model_info())

    # Auto-tuning methods
    def autotune_for_model(self, **kwargs) -> Dict[str, Any]:
        return self._run(self._async_client.autotune_for_model(**kwargs))

    def get_autotune_history(self, **kwargs) -> List[Dict[str, Any]]:
        return self._run(self._async_client.get_autotune_history(**kwargs))

    def get_autotuner_info(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_autotuner_info())

    def close(self):
        self._run(self._async_client.close())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def example_usage():
    """Example: How to use AutonomousClient"""

    async with AutonomousClient(verbose=True) as client:
        # Check health
        healthy = await client.health_check()
        print(f"System healthy: {healthy}")

        # Get resources
        resources = await client.get_resources()
        print(f"VRAM: {resources['gpu_memory_percent']:.1f}%")
        print(f"Temp: {resources['gpu_temperature']}°C")

        # Request device decision
        decision = await client.decide_device(
            task_type="training",
            estimated_vram=3.0,
            priority="high"
        )
        print(f"Device: {decision['device']}")
        print(f"Reason: {decision['reason']}")

        # Check if ready for training
        ready = await client.is_ready_for_training(estimated_vram=4.0)
        print(f"Ready for training: {ready}")

        if ready:
            # Automated training
            result = await client.auto_train(
                model_name="TestModel",
                epochs=10,
                estimated_vram=4.0
            )

            if result['success']:
                print(f"Training started: {result['session']['session_id']}")

                # Monitor progress
                while True:
                    status = await client.get_training_status()
                    if status is None:
                        break

                    print(f"Progress: {status['progress']:.1f}%")
                    await asyncio.sleep(5)

                # Complete
                await client.complete_training(success=True)
                print("Training completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run example
    asyncio.run(example_usage())
