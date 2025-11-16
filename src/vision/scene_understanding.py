"""
Scene Understanding Enhancement - فهم السياق البصري المحسّن
=============================================================

يوسّع ImageAnalyzer بقدرات فهم السياق والعلاقات المكانية

Part of Deep Cognition v1.2 Lite - Scene Understanding
Addresses Q7 (Scene context interpretation)

Author: Noogh AI Team
Date: 2025-11-10
Priority: HIGH
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import os

# استيراد ImageAnalyzer الحالي
try:
    from vision.image_analyzer import (
        ImageAnalyzer,
        ImageAnalysisResult,
        ImageType,
        ColorScheme
    )
    IMAGE_ANALYZER_AVAILABLE = True
except Exception as e:
    # Error caught: {e}
    IMAGE_ANALYZER_AVAILABLE = False


class SceneType(str, Enum):
    """أنواع المشاهد المُحسّنة"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    URBAN = "urban"
    NATURE = "nature"
    WORKSPACE = "workspace"
    SOCIAL = "social"
    TRANSPORTATION = "transportation"
    UNKNOWN = "unknown"


class SpatialRelation(str, Enum):
    """العلاقات المكانية"""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    OUTSIDE = "outside"
    NEAR = "near"
    FAR = "far"
    CENTERED = "centered"


class LightingCondition(str, Enum):
    """ظروف الإضاءة"""
    BRIGHT = "bright"
    DIM = "dim"
    NATURAL_DAYLIGHT = "natural_daylight"
    ARTIFICIAL = "artificial"
    MIXED = "mixed"
    BACKLIT = "backlit"
    DARK = "dark"


@dataclass
class SpatialHint:
    """تلميح مكاني"""
    region: str  # "top", "bottom", "center", etc.
    density: float  # 0.0-1.0 (كثافة المحتوى في المنطقة)
    dominant_feature: str  # الميزة السائدة


@dataclass
class SceneContext:
    """سياق المشهد"""
    scene_type: SceneType
    lighting_condition: LightingCondition
    time_of_day_hint: Optional[str]  # "morning", "noon", "evening", "night"
    weather_hint: Optional[str]  # "sunny", "cloudy", "rainy" (من الألوان)
    spatial_layout: Dict[str, SpatialHint]
    contextual_clues: List[str]  # قرائن سياقية مستنتجة
    confidence: float  # 0.0-1.0


@dataclass
class EnhancedSceneAnalysis:
    """تحليل محسّن للمشهد"""
    basic_analysis: Any  # ImageAnalysisResult from ImageAnalyzer
    scene_context: SceneContext
    complexity_score: float  # 0.0-1.0
    interpretability: float  # مدى سهولة تفسير المشهد
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_context.scene_type.value,
            "lighting": self.scene_context.lighting_condition.value,
            "time_hint": self.scene_context.time_of_day_hint,
            "weather_hint": self.scene_context.weather_hint,
            "contextual_clues": self.scene_context.contextual_clues,
            "complexity": self.complexity_score,
            "interpretability": self.interpretability,
            "confidence": self.scene_context.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class SceneUnderstandingEngine:
    """
    محرك فهم السياق البصري

    يوسّع ImageAnalyzer بقدرات:
    1. تصنيف نوع المشهد (indoor/outdoor/urban/nature)
    2. تحليل ظروف الإضاءة (bright/dim/natural)
    3. استنتاج الوقت من اليوم (morning/evening/night)
    4. تحليل العلاقات المكانية
    5. استخراج قرائن سياقية
    """

    def __init__(self):
        # تحميل ImageAnalyzer الأساسي
        self.image_analyzer = ImageAnalyzer() if IMAGE_ANALYZER_AVAILABLE else None

        # إحصائيات
        self.total_analyzed = 0
        self.scene_type_counts: Dict[SceneType, int] = {}

    def analyze_scene(self, image_path: str) -> EnhancedSceneAnalysis:
        """
        تحليل مشهد شامل

        Args:
            image_path: مسار الصورة

        Returns:
            EnhancedSceneAnalysis
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # تحليل أساسي من ImageAnalyzer
        basic_analysis = None
        if self.image_analyzer:
            basic_analysis = self.image_analyzer.analyze_image(image_path)

        # تحليل السياق
        scene_context = self._analyze_context(basic_analysis) if basic_analysis else self._fallback_context()

        # حساب التعقيد والقابلية للتفسير
        complexity = self._calculate_complexity(basic_analysis) if basic_analysis else 0.5
        interpretability = self._calculate_interpretability(scene_context, complexity)

        result = EnhancedSceneAnalysis(
            basic_analysis=basic_analysis,
            scene_context=scene_context,
            complexity_score=complexity,
            interpretability=interpretability
        )

        self.total_analyzed += 1
        self.scene_type_counts[scene_context.scene_type] = \
            self.scene_type_counts.get(scene_context.scene_type, 0) + 1

        return result

    def _analyze_context(self, basic: 'ImageAnalysisResult') -> SceneContext:
        """تحليل السياق من النتائج الأساسية"""

        # 1. تحديد نوع المشهد
        scene_type = self._determine_scene_type(basic)

        # 2. تحليل الإضاءة
        lighting = self._analyze_lighting(basic)

        # 3. استنتاج الوقت من اليوم
        time_hint = self._infer_time_of_day(basic, lighting)

        # 4. استنتاج الطقس
        weather_hint = self._infer_weather(basic)

        # 5. تحليل التوزيع المكاني
        spatial_layout = self._analyze_spatial_layout(basic)

        # 6. استخراج قرائن سياقية
        clues = self._extract_contextual_clues(
            scene_type, lighting, time_hint, weather_hint, spatial_layout
        )

        # 7. حساب الثقة
        confidence = self._calculate_context_confidence(basic, scene_type, lighting)

        return SceneContext(
            scene_type=scene_type,
            lighting_condition=lighting,
            time_of_day_hint=time_hint,
            weather_hint=weather_hint,
            spatial_layout=spatial_layout,
            contextual_clues=clues,
            confidence=confidence
        )

    def _determine_scene_type(self, basic: 'ImageAnalysisResult') -> SceneType:
        """تحديد نوع المشهد"""
        # قواعد بسيطة بناءً على خصائص الصورة

        image_type = basic.image_type
        color_scheme = basic.colors.color_scheme
        brightness = basic.colors.brightness_avg
        complexity = basic.content.estimated_complexity

        # DOCUMENT/SCREENSHOT → WORKSPACE
        if image_type in ["document", "screenshot"]:
            return SceneType.WORKSPACE

        # DIAGRAM → WORKSPACE
        if image_type == "diagram":
            return SceneType.WORKSPACE

        # ألوان زاهية + تعقيد عالي → OUTDOOR/NATURE
        if color_scheme == "vibrant" and complexity > 0.6:
            # سطوع عالي → OUTDOOR
            if brightness > 150:
                return SceneType.OUTDOOR
            else:
                return SceneType.NATURE

        # ألوان داكنة → INDOOR
        if color_scheme == "dark" or brightness < 100:
            return SceneType.INDOOR

        # PHOTO مع تعقيد متوسط → SOCIAL
        if image_type == "photo" and 0.3 < complexity < 0.6:
            return SceneType.SOCIAL

        # افتراضي
        return SceneType.UNKNOWN

    def _analyze_lighting(self, basic: 'ImageAnalysisResult') -> LightingCondition:
        """تحليل ظروف الإضاءة"""
        brightness = basic.colors.brightness_avg
        color_diversity = basic.colors.color_diversity

        # سطوع عالي جداً → BRIGHT
        if brightness > 200:
            return LightingCondition.BRIGHT

        # سطوع منخفض → DIM أو DARK
        if brightness < 80:
            return LightingCondition.DARK
        elif brightness < 120:
            return LightingCondition.DIM

        # تنوع ألوان عالي + سطوع متوسط → NATURAL_DAYLIGHT
        if color_diversity > 0.4 and 120 < brightness < 180:
            return LightingCondition.NATURAL_DAYLIGHT

        # سطوع متوسط + تنوع منخفض → ARTIFICIAL
        if color_diversity < 0.3 and 100 < brightness < 160:
            return LightingCondition.ARTIFICIAL

        # الباقي → MIXED
        return LightingCondition.MIXED

    def _infer_time_of_day(self, basic: 'ImageAnalysisResult',
                          lighting: LightingCondition) -> Optional[str]:
        """استنتاج الوقت من اليوم"""
        brightness = basic.colors.brightness_avg

        # BRIGHT + NATURAL → NOON
        if lighting == LightingCondition.NATURAL_DAYLIGHT and brightness > 180:
            return "noon"

        # NATURAL + متوسط → MORNING أو AFTERNOON
        if lighting == LightingCondition.NATURAL_DAYLIGHT:
            return "morning" if brightness < 150 else "afternoon"

        # DIM/DARK → EVENING أو NIGHT
        if lighting in [LightingCondition.DIM, LightingCondition.DARK]:
            return "evening" if brightness > 60 else "night"

        return None

    def _infer_weather(self, basic: 'ImageAnalysisResult') -> Optional[str]:
        """استنتاج الطقس من الألوان"""
        color_scheme = basic.colors.color_scheme
        brightness = basic.colors.brightness_avg

        # VIBRANT + BRIGHT → SUNNY
        if color_scheme == "vibrant" and brightness > 170:
            return "sunny"

        # LIGHT + تنوع منخفض → CLOUDY
        if color_scheme == "light" and basic.colors.color_diversity < 0.3:
            return "cloudy"

        # DARK + سطوع منخفض → RAINY
        if color_scheme == "dark" and brightness < 100:
            return "rainy"

        return None

    def _analyze_spatial_layout(self, basic: 'ImageAnalysisResult') -> Dict[str, SpatialHint]:
        """تحليل التوزيع المكاني (تبسيط)"""
        # في النظام الكامل: تقسيم الصورة إلى شبكة وتحليل كل منطقة
        # الآن: تحليل بسيط بناءً على خصائص عامة

        width = basic.dimensions.width
        height = basic.dimensions.height
        complexity = basic.content.estimated_complexity

        layout = {}

        # المنطقة العلوية (عادة سماء في outdoor، سقف في indoor)
        layout["top"] = SpatialHint(
            region="top",
            density=0.3,  # عادة أقل كثافة
            dominant_feature="sky_or_ceiling"
        )

        # المنطقة الوسطى (المحتوى الرئيسي)
        layout["center"] = SpatialHint(
            region="center",
            density=complexity,  # أعلى كثافة
            dominant_feature="main_content"
        )

        # المنطقة السفلية (عادة أرض/قاعدة)
        layout["bottom"] = SpatialHint(
            region="bottom",
            density=0.5,
            dominant_feature="ground_or_base"
        )

        return layout

    def _extract_contextual_clues(self,
                                  scene_type: SceneType,
                                  lighting: LightingCondition,
                                  time_hint: Optional[str],
                                  weather_hint: Optional[str],
                                  spatial_layout: Dict[str, SpatialHint]) -> List[str]:
        """استخراج قرائن سياقية"""
        clues = []

        # قرائن من نوع المشهد
        if scene_type == SceneType.OUTDOOR:
            clues.append("Open environment, likely natural or urban setting")
        elif scene_type == SceneType.INDOOR:
            clues.append("Enclosed space, possibly building interior")
        elif scene_type == SceneType.WORKSPACE:
            clues.append("Work or study environment")
        elif scene_type == SceneType.SOCIAL:
            clues.append("Social gathering or event")

        # قرائن من الإضاءة والوقت
        if time_hint:
            clues.append(f"Time of day appears to be {time_hint}")

        if weather_hint:
            clues.append(f"Weather conditions suggest {weather_hint}")

        # قرائن من التوزيع المكاني
        center_density = spatial_layout.get("center", SpatialHint("center", 0.5, "unknown")).density
        if center_density > 0.7:
            clues.append("High activity or detail in central region")
        elif center_density < 0.3:
            clues.append("Minimal central content, possibly minimalist composition")

        return clues

    def _calculate_complexity(self, basic: 'ImageAnalysisResult') -> float:
        """حساب تعقيد المشهد"""
        if not basic:
            return 0.5

        # دمج عدة عوامل
        edge_ratio = min(basic.content.edges_detected / (basic.dimensions.width * basic.dimensions.height), 1.0) if basic.dimensions.width * basic.dimensions.height > 0 else 0.5
        color_diversity = basic.colors.color_diversity
        content_complexity = basic.content.estimated_complexity

        # متوسط موزون
        complexity = (edge_ratio * 0.4 + color_diversity * 0.3 + content_complexity * 0.3)

        return min(complexity, 1.0)

    def _calculate_interpretability(self, context: SceneContext, complexity: float) -> float:
        """حساب مدى سهولة تفسير المشهد"""
        # المشاهد البسيطة أسهل للتفسير
        base_interpretability = 1.0 - (complexity * 0.5)

        # المشاهد المعروفة أسهل للتفسير
        if context.scene_type != SceneType.UNKNOWN:
            base_interpretability += 0.2

        # وجود قرائن سياقية يسهل التفسير
        clue_bonus = min(len(context.contextual_clues) * 0.05, 0.2)
        base_interpretability += clue_bonus

        return min(base_interpretability, 1.0)

    def _calculate_context_confidence(self,
                                     basic: 'ImageAnalysisResult',
                                     scene_type: SceneType,
                                     lighting: LightingCondition) -> float:
        """حساب ثقة تحليل السياق"""
        confidence = 0.5  # قيمة أساسية

        # مؤشرات واضحة → ثقة أعلى
        if scene_type != SceneType.UNKNOWN:
            confidence += 0.2

        if lighting != LightingCondition.MIXED:
            confidence += 0.15

        # جودة الصورة تؤثر على الثقة
        if basic.dimensions.megapixels > 1.0:
            confidence += 0.1

        # تنوع الألوان يساعد على التمييز
        if basic.colors.color_diversity > 0.4:
            confidence += 0.05

        return min(confidence, 1.0)

    def _fallback_context(self) -> SceneContext:
        """سياق احتياطي عند عدم توفر ImageAnalyzer"""
        return SceneContext(
            scene_type=SceneType.UNKNOWN,
            lighting_condition=LightingCondition.MIXED,
            time_of_day_hint=None,
            weather_hint=None,
            spatial_layout={},
            contextual_clues=["Basic analysis unavailable"],
            confidence=0.1
        )

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات المحرك"""
        return {
            "total_analyzed": self.total_analyzed,
            "scene_type_distribution": {
                scene_type.value: count
                for scene_type, count in self.scene_type_counts.items()
            },
            "image_analyzer_available": IMAGE_ANALYZER_AVAILABLE
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🔍 Scene Understanding Engine - Test")
    print("=" * 70)
    print()

    engine = SceneUnderstandingEngine()

    print(f"ImageAnalyzer Available: {engine.image_analyzer is not None}")
    print()

    # في الاستخدام الحقيقي:
    # result = engine.analyze_scene("path/to/image.jpg")
    # print(f"Scene Type: {result.scene_context.scene_type.value}")
    # print(f"Lighting: {result.scene_context.lighting_condition.value}")
    # print(f"Contextual Clues: {result.scene_context.contextual_clues}")

    print("=" * 70)
    print("✅ Scene Understanding Engine initialized!")
    print("   Enhanced capabilities:")
    print("   - Scene type classification (indoor/outdoor/urban/nature/...)")
    print("   - Lighting condition analysis")
    print("   - Time of day inference")
    print("   - Weather hints from colors")
    print("   - Spatial layout analysis")
    print("   - Contextual clue extraction")
    print()
