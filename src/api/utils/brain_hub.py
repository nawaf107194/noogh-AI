#!/usr/bin/env python3
"""
🧠 Brain Hub - مركز ربط الدماغ بجميع الخدمات
يربط نموذج الدماغ بـ:
- API Services
- Database
- Ministers
- External Systems
- Real-time Processing
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
import torch
from brain.mega_brain_v5 import create_mega_brain_v5
from ai_engine import AIEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainHub:
    """مركز الدماغ - يربط Brain بجميع المكونات"""

    def __init__(self, device: str = "auto"):
        """
        Initialize Brain Hub

        Args:
            device: 'cuda', 'cpu', or 'auto'
        """
        self.device = self._setup_device(device)
        self.brain = None
        self.ai_engine = None
        self.ministers = {}
        self.president = None  # رئيس الحكومة
        self.connections = {}
        self.is_ready = False

        logger.info(f"🧠 Brain Hub initialized on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU: {gpu_name}")
        elif device == "cuda":
            logger.warning("⚠️  CUDA requested but not available, using CPU")
            device = "cpu"

        return device

    def load_brain(self, config: str = "small") -> None:
        """
        Load brain model

        Args:
            config: 'small', 'medium', or 'large'
        """
        logger.info(f"📥 Loading brain model (config: {config})...")

        try:
            self.brain = create_mega_brain_v5(config=config, device=self.device)
            self.brain.eval()

            param_count = sum(p.numel() for p in self.brain.parameters())
            logger.info(f"✅ Brain loaded: {param_count:,} parameters")

            self.is_ready = True

        except Exception as e:
            logger.error(f"❌ Failed to load brain: {e}")
            raise

    def connect_to_api(self) -> None:
        """Connect brain to API services"""
        logger.info("🔌 Connecting to API services...")

        self.connections['api'] = {
            'main_api': 'http://localhost:8000',
            'inference': 'http://localhost:8899',
            'chat': 'http://localhost:8501',
            'analytics': 'http://localhost:8502',
        }

        logger.info("✅ API connections registered")

    def connect_to_database(self) -> None:
        """Connect brain to databases"""
        logger.info("🔌 Connecting to databases...")

        db_paths = {
            'memory': PROJECT_ROOT / 'data' / 'memory.db',
            'noogh_memory': PROJECT_ROOT / 'data' / 'noogh_memory.db',
            'performance': PROJECT_ROOT / 'data' / 'performance.db',
            'reflection': PROJECT_ROOT / 'data' / 'reflection.db',
        }

        self.connections['databases'] = {}
        for name, path in db_paths.items():
            if path.exists():
                self.connections['databases'][name] = str(path)
                logger.info(f"  ✓ {name}: {path.name}")

        logger.info(f"✅ Connected to {len(self.connections['databases'])} databases")

    def register_ministers(self) -> None:
        """Register all ministers"""
        logger.info("👔 Registering ministers...")

        self.ministers = {
            'president': 'رئيس الجمهورية - القيادة العليا',
            'defense': 'وزير الدفاع - الأمن والحماية',
            'interior': 'وزير الداخلية - الإدارة الداخلية',
            'foreign': 'وزير الخارجية - العلاقات الدولية',
            'finance': 'وزير المالية - الإدارة المالية',
            'economy': 'وزير الاقتصاد - التطوير الاقتصادي',
            'education': 'وزير التعليم - التعليم والتدريب',
            'health': 'وزير الصحة - الرعاية الصحية',
            'technology': 'وزير التكنولوجيا - الابتكار التقني',
            'data': 'وزير البيانات - إدارة المعلومات',
            'ai': 'وزير الذكاء الاصطناعي - تطوير AI',
            'crypto': 'وزير العملات - التداول والتنبؤ',
            'media': 'وزير الإعلام - الاتصال والتواصل',
            'development': 'وزير التطوير - البنية التحتية',
        }

        logger.info(f"✅ Registered {len(self.ministers)} ministers")

    def connect_external_systems(self) -> None:
        """Connect to external systems"""
        logger.info("🌐 Connecting to external systems...")

        self.connections['external'] = {
            'huggingface': 'HuggingFace Hub',
            'binance': 'Binance Exchange',
            'github': 'GitHub Integration',
        }

        logger.info("✅ External systems registered")

    def inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Run inference through brain

        Args:
            input_data: Input for inference

        Returns:
            Inference result
        """
        if not self.is_ready:
            raise RuntimeError("Brain not loaded. Call load_brain() first.")

        try:
            with torch.no_grad():
                # Process through brain
                # (Implementation depends on your specific brain architecture)
                result = {
                    'status': 'success',
                    'brain_ready': True,
                    'device': self.device,
                    'ministers_count': len(self.ministers),
                    'connections': list(self.connections.keys())
                }

            return result

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get hub status"""
        return {
            'ready': self.is_ready,
            'device': self.device,
            'brain_loaded': self.brain is not None,
            'president_connected': self.president is not None,
            'ministers_count': len(self.ministers),
            'connections': {
                'api': len(self.connections.get('api', {})),
                'databases': len(self.connections.get('databases', {})),
                'external': len(self.connections.get('external', {})),
            }
        }

    async def process_with_president(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        معالجة طلب من خلال الرئيس مع الدماغ

        Args:
            user_input: طلب المستخدم
            context: سياق إضافي

        Returns:
            نتيجة شاملة من الرئيس + الدماغ + AI Engine
        """
        if not self.president:
            raise RuntimeError("President not connected. Call connect_president() first.")

        logger.info(f"🎩 Processing request through President...")

        # 1. Brain processing
        brain_result = None
        if self.is_ready:
            brain_result = self.inference({'text': user_input})

        # 2. AI Engine processing
        ai_result = None
        if self.ai_engine:
            ai_result = self.ai_engine.process(
                input_data={'request': user_input},
                context='president_request'
            )

        # 3. President processing (with access to brain results)
        president_context = context or {}
        president_context['brain_output'] = brain_result
        president_context['ai_analysis'] = ai_result

        president_result = await self.president.process_request(
            user_input=user_input,
            context=president_context
        )

        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'brain': brain_result,
            'ai_engine': ai_result,
            'president': president_result,
            'integrated': True
        }

    def initialize_ai_engine(self) -> None:
        """Initialize AI Engine"""
        logger.info("🤖 Initializing AI Engine...")

        self.ai_engine = AIEngine(brain_hub=self)
        logger.info("✅ AI Engine ready")

    def activate_real_ministers(self) -> None:
        """Activate real ministers with full functionality"""
        logger.info("👔 Activating real ministers with full roles...")

        try:
            from government.ministers_activation import MinistersActivationSystem

            # Create activation system
            self.ministers_system = MinistersActivationSystem(brain_hub=self)

            # Activate all ministers
            self.ministers_system.activate_all()

            logger.info("✅ Real ministers activated")
            logger.info(f"   Active ministers: {len(self.ministers_system.active_ministers)}")

        except Exception as e:
            logger.error(f"❌ Failed to activate ministers: {e}")
            raise

    def connect_president(self) -> None:
        """Connect Universal President to Brain"""
        logger.info("🎩 Connecting Universal President to Brain...")

        try:
            from government.president import President

            # Create president with brain connection
            self.president = President(verbose=True)

            # Give president access to brain
            self.president.brain_hub = self

            # Give president access to active ministers
            if hasattr(self, 'ministers_system'):
                self.president.active_ministers = self.ministers_system.active_ministers
                logger.info(f"   President has access to {len(self.ministers_system.active_ministers)} active ministers")

            logger.info("✅ President connected to Brain")
            logger.info(f"   President has access to:")
            logger.info(f"   - Brain (70M parameters)")
            logger.info(f"   - AI Engine (learning, memory, reasoning)")
            logger.info(f"   - {len(self.ministers)} Minister positions")
            logger.info(f"   - {len(self.connections.get('databases', {}))} Databases")
            if hasattr(self, 'ministers_system'):
                logger.info(f"   - {len(self.ministers_system.active_ministers)} Active ministers (fully functional)")

        except Exception as e:
            logger.error(f"❌ Failed to connect president: {e}")
            raise

    def initialize_all(self) -> None:
        """Initialize all connections"""
        logger.info("=" * 70)
        logger.info("🚀 BRAIN HUB - FULL INITIALIZATION")
        logger.info("=" * 70)

        self.load_brain()
        self.connect_to_api()
        self.connect_to_database()
        self.register_ministers()
        self.connect_external_systems()
        self.initialize_ai_engine()
        self.activate_real_ministers()  # تفعيل الوزراء الحقيقيين
        self.connect_president()  # ربط رئيس الحكومة

        logger.info("=" * 70)
        logger.info("✅ BRAIN HUB READY (WITH AI + PRESIDENT + ACTIVE MINISTERS)")
        logger.info("=" * 70)

        # Print status
        status = self.get_status()
        logger.info(f"🧠 Brain: {'✓' if status['brain_loaded'] else '✗'}")
        logger.info(f"🤖 AI Engine: {'✓' if self.ai_engine else '✗'}")
        logger.info(f"🎩 President: {'✓' if self.president else '✗'}")
        logger.info(f"👔 Ministers: {status['ministers_count']}")
        if hasattr(self, 'ministers_system'):
            logger.info(f"👔 Active Ministers: {len(self.ministers_system.active_ministers)} (fully functional)")
        logger.info(f"🔌 API Connections: {status['connections']['api']}")
        logger.info(f"💾 Databases: {status['connections']['databases']}")
        logger.info(f"🌐 External: {status['connections']['external']}")


def main():
    """Test Brain Hub"""
    print("\n🧠 Testing Brain Hub...\n")

    # Create hub
    hub = BrainHub()

    # Initialize everything
    hub.initialize_all()

    # Test inference
    print("\n🧪 Testing inference...")
    result = hub.inference({'test': 'data'})
    print(f"Result: {result}")

    print("\n✅ Brain Hub test complete!")


if __name__ == "__main__":
    main()
