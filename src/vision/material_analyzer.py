"""
Material Analysis Enhancement - تحليل المواد والإضاءة
=======================================================

يميز بين تغيّر الإضاءة وتغيّر المادة في الصور

Part of Deep Cognition v1.2 Lite - Material Understanding
Addresses Q1 (Lighting vs Material differentiation)

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
        ImageAnalysisResult
    )
    IMAGE_ANALYZER_AVAILABLE = True
except Exception as e:
    # Error caught: {e}
    IMAGE_ANALYZER_AVAILABLE = False


class MaterialType(str, Enum):
    """أنواع المواد"""
    METAL = "metal"
    WOOD = "wood"
    PLASTIC = "plastic"
    FABRIC = "fabric"
    GLASS = "glass"
    PAPER = "paper"
    STONE = "stone"
    SKIN = "skin"
    UNKNOWN = "unknown"


class SurfaceProperty(str, Enum):
    """خصائص السطح"""
    GLOSSY = "glossy"  # لامع/عاكس
    MATTE = "matte"  # مطفي
    SEMI_GLOSS = "semi_gloss"  # شبه لامع
    ROUGH = "rough"  # خشن
    SMOOTH = "smooth"  # ناعم
    TEXTURED = "textured"  # محبب


class LightingEffect(str, Enum):
    """تأثيرات الإضاءة"""
    DIRECT_LIGHT = "direct_light"  # إضاءة مباشرة
    DIFFUSED_LIGHT = "diffused_light"  # إضاءة منتشرة
    SHADOW = "shadow"  # ظل
    HIGHLIGHT = "highlight"  # انعكاس قوي
    AMBIENT = "ambient"  # إضاءة محيطة
    BACKLIGHT = "backlight"  # إضاءة خلفية


@dataclass
class MaterialProperties:
    """خصائص المادة"""
    material_type: MaterialType
    surface_property: SurfaceProperty
    reflectivity: float  # 0.0-1.0 (مطفي → عاكس)
    texture_roughness: float  # 0.0-1.0 (ناعم → خشن)
    color_uniformity: float  # 0.0-1.0 (متنوع → موحد)
    confidence: float  # 0.0-1.0


@dataclass
class LightingAnalysis:
    """تحليل الإضاءة"""
    dominant_effect: LightingEffect
    intensity: float  # 0.0-1.0
    direction_hint: Optional[str]  # "top", "side", "front", etc.
    color_temperature: str  # "warm", "cool", "neutral"
    uniformity: float  # 0.0-1.0 (متجانس → غير متجانس)


@dataclass
class LightingMaterialSeparation:
    """فصل بين الإضاءة والمادة"""
    material: MaterialProperties
    lighting: LightingAnalysis
    separation_confidence: float  # مدى ثقة الفصل
    is_lighting_dominant: bool  # هل التغيير بسبب الإضاءة؟
    is_material_dominant: bool  # هل التغيير بسبب المادة؟
    explanation: str  # تفسير للتحليل
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material_type": self.material.material_type.value,
            "surface_property": self.material.surface_property.value,
            "reflectivity": self.material.reflectivity,
            "lighting_effect": self.lighting.dominant_effect.value,
            "lighting_intensity": self.lighting.intensity,
            "separation_confidence": self.separation_confidence,
            "lighting_dominant": self.is_lighting_dominant,
            "material_dominant": self.is_material_dominant,
            "explanation": self.explanation
        }


class MaterialAnalyzer:
    """
    محلل المواد والإضاءة

    يفصل بين:
    1. تغيّر الإضاءة (Lighting change)
    2. تغيّر المادة (Material change)

    عبر تحليل:
    - Reflectivity (انعكاسية السطح)
    - Texture (خشونة/نعومة)
    - Color uniformity (توحيد اللون)
    - Lighting patterns (أنماط الإضاءة)
    """

    def __init__(self):
        # تحميل ImageAnalyzer الأساسي
        self.image_analyzer = ImageAnalyzer() if IMAGE_ANALYZER_AVAILABLE else None

        # إحصائيات
        self.total_analyzed = 0
        self.material_type_counts: Dict[MaterialType, int] = {}

    def analyze(self, image_path: str) -> LightingMaterialSeparation:
        """
        تحليل شامل لفصل الإضاءة عن المادة

        Args:
            image_path: مسار الصورة

        Returns:
            LightingMaterialSeparation
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # تحليل أساسي
        basic_analysis = None
        if self.image_analyzer:
            basic_analysis = self.image_analyzer.analyze_image(image_path)

        # تحليل المادة
        material = self._analyze_material(basic_analysis) if basic_analysis else self._fallback_material()

        # تحليل الإضاءة
        lighting = self._analyze_lighting(basic_analysis) if basic_analysis else self._fallback_lighting()

        # فصل بين الإضاءة والمادة
        separation = self._separate_lighting_material(material, lighting, basic_analysis)

        self.total_analyzed += 1
        self.material_type_counts[material.material_type] = \
            self.material_type_counts.get(material.material_type, 0) + 1

        return separation

    def _analyze_material(self, basic: 'ImageAnalysisResult') -> MaterialProperties:
        """تحليل خصائص المادة"""

        # 1. تحديد نوع المادة
        material_type = self._determine_material_type(basic)

        # 2. تحليل خصائص السطح
        surface_property = self._analyze_surface(basic)

        # 3. حساب الانعكاسية
        reflectivity = self._calculate_reflectivity(basic)

        # 4. حساب خشونة الـ texture
        texture_roughness = self._calculate_texture_roughness(basic)

        # 5. حساب توحيد اللون
        color_uniformity = self._calculate_color_uniformity(basic)

        # 6. حساب الثقة
        confidence = self._calculate_material_confidence(
            material_type, surface_property, basic
        )

        return MaterialProperties(
            material_type=material_type,
            surface_property=surface_property,
            reflectivity=reflectivity,
            texture_roughness=texture_roughness,
            color_uniformity=color_uniformity,
            confidence=confidence
        )

    def _determine_material_type(self, basic: 'ImageAnalysisResult') -> MaterialType:
        """تحديد نوع المادة من خصائص الصورة"""
        # قواعد تقريبية بناءً على خصائص بصرية

        color_scheme = basic.colors.color_scheme
        brightness = basic.colors.brightness_avg
        color_diversity = basic.colors.color_diversity
        complexity = basic.content.estimated_complexity

        # METAL: سطوع عالي + تنوع منخفض + حواف حادة
        if brightness > 180 and color_diversity < 0.2 and complexity > 0.6:
            return MaterialType.METAL

        # GLASS: سطوع متوسط-عالي + شفافية (تعقيد منخفض)
        if 140 < brightness < 200 and complexity < 0.3:
            return MaterialType.GLASS

        # WOOD: ألوان دافئة + تعقيد متوسط
        if 80 < brightness < 150 and 0.3 < complexity < 0.6:
            # افتراض: dominant color في نطاق البني
            return MaterialType.WOOD

        # FABRIC: تعقيد عالي (texture) + تنوع لوني
        if complexity > 0.7 and color_diversity > 0.4:
            return MaterialType.FABRIC

        # PAPER: سطوع عالي + تنوع منخفض + تعقيد منخفض
        if brightness > 170 and color_diversity < 0.3 and complexity < 0.4:
            return MaterialType.PAPER

        # PLASTIC: سطوع متوسط + ألوان موحدة
        if 100 < brightness < 170 and color_diversity < 0.35:
            return MaterialType.PLASTIC

        return MaterialType.UNKNOWN

    def _analyze_surface(self, basic: 'ImageAnalysisResult') -> SurfaceProperty:
        """تحليل خصائص السطح"""
        brightness = basic.colors.brightness_avg
        complexity = basic.content.estimated_complexity

        # GLOSSY: سطوع عالي + تعقيد منخفض
        if brightness > 170 and complexity < 0.4:
            return SurfaceProperty.GLOSSY

        # ROUGH: تعقيد عالي
        if complexity > 0.7:
            return SurfaceProperty.ROUGH

        # SMOOTH: تعقيد منخفض
        if complexity < 0.3:
            return SurfaceProperty.SMOOTH

        # MATTE: سطوع متوسط-منخفض + تعقيد متوسط
        if brightness < 130:
            return SurfaceProperty.MATTE

        return SurfaceProperty.SEMI_GLOSS

    def _calculate_reflectivity(self, basic: 'ImageAnalysisResult') -> float:
        """حساب الانعكاسية"""
        # السطوح العاكسة: سطوع عالي + تنوع لوني منخفض
        brightness_factor = basic.colors.brightness_avg / 255.0
        uniformity_factor = 1.0 - basic.colors.color_diversity

        reflectivity = (brightness_factor * 0.7 + uniformity_factor * 0.3)
        return min(reflectivity, 1.0)

    def _calculate_texture_roughness(self, basic: 'ImageAnalysisResult') -> float:
        """حساب خشونة الـ texture"""
        # الخشونة تظهر في تعقيد الحواف
        edge_ratio = basic.content.edges_detected / (basic.dimensions.width * basic.dimensions.height) if basic.dimensions.width * basic.dimensions.height > 0 else 0

        roughness = min(edge_ratio * 5, 1.0)  # normalization
        return roughness

    def _calculate_color_uniformity(self, basic: 'ImageAnalysisResult') -> float:
        """حساب توحيد اللون"""
        # عكس التنوع اللوني
        uniformity = 1.0 - basic.colors.color_diversity
        return uniformity

    def _calculate_material_confidence(self,
                                      material_type: MaterialType,
                                      surface_property: SurfaceProperty,
                                      basic: 'ImageAnalysisResult') -> float:
        """حساب ثقة تحديد المادة"""
        confidence = 0.5  # قيمة أساسية

        # أنواع محددة → ثقة أعلى
        if material_type != MaterialType.UNKNOWN:
            confidence += 0.2

        # خصائص سطح واضحة → ثقة أعلى
        if surface_property in [SurfaceProperty.GLOSSY, SurfaceProperty.ROUGH]:
            confidence += 0.15

        # جودة صورة عالية → ثقة أعلى
        if basic.dimensions.megapixels > 1.0:
            confidence += 0.1

        return min(confidence, 1.0)

    def _analyze_lighting(self, basic: 'ImageAnalysisResult') -> LightingAnalysis:
        """تحليل الإضاءة"""

        # 1. تحديد التأثير السائد
        dominant_effect = self._determine_lighting_effect(basic)

        # 2. حساب الشدة
        intensity = basic.colors.brightness_avg / 255.0

        # 3. تلميح للاتجاه (بسيط)
        direction_hint = self._infer_lighting_direction(basic)

        # 4. درجة حرارة اللون
        color_temp = self._determine_color_temperature(basic)

        # 5. توحيد الإضاءة
        uniformity = self._calculate_lighting_uniformity(basic)

        return LightingAnalysis(
            dominant_effect=dominant_effect,
            intensity=intensity,
            direction_hint=direction_hint,
            color_temperature=color_temp,
            uniformity=uniformity
        )

    def _determine_lighting_effect(self, basic: 'ImageAnalysisResult') -> LightingEffect:
        """تحديد تأثير الإضاءة السائد"""
        brightness = basic.colors.brightness_avg

        # إضاءة ساطعة جداً → DIRECT or HIGHLIGHT
        if brightness > 200:
            return LightingEffect.HIGHLIGHT

        # إضاءة منخفضة → SHADOW
        if brightness < 80:
            return LightingEffect.SHADOW

        # إضاءة متوسطة موحدة → AMBIENT أو DIFFUSED
        if 100 < brightness < 170:
            return LightingEffect.AMBIENT

        return LightingEffect.DIFFUSED_LIGHT

    def _infer_lighting_direction(self, basic: 'ImageAnalysisResult') -> Optional[str]:
        """استنتاج اتجاه الإضاءة (تبسيط)"""
        # في النظام الكامل: تحليل gradient للسطوع
        # الآن: تخمين بسيط

        brightness = basic.colors.brightness_avg

        if brightness > 180:
            return "front"  # إضاءة أمامية
        elif brightness < 100:
            return "back"  # backlight محتمل
        else:
            return "top"  # افتراضي

    def _determine_color_temperature(self, basic: 'ImageAnalysisResult') -> str:
        """تحديد درجة حرارة اللون"""
        # في النظام الكامل: تحليل dominant color hue
        # الآن: من color_scheme

        color_scheme = basic.colors.color_scheme

        if color_scheme in ["vibrant", "colorful"]:
            return "neutral"
        elif color_scheme == "light":
            return "cool"
        elif color_scheme == "dark":
            return "warm"

        return "neutral"

    def _calculate_lighting_uniformity(self, basic: 'ImageAnalysisResult') -> float:
        """حساب توحيد الإضاءة"""
        # الإضاءة الموحدة: تنوع لوني منخفض + تعقيد منخفض
        color_factor = 1.0 - basic.colors.color_diversity
        complexity_factor = 1.0 - basic.content.estimated_complexity

        uniformity = (color_factor * 0.6 + complexity_factor * 0.4)
        return min(uniformity, 1.0)

    def _separate_lighting_material(self,
                                   material: MaterialProperties,
                                   lighting: LightingAnalysis,
                                   basic: Optional['ImageAnalysisResult']) -> LightingMaterialSeparation:
        """الفصل النهائي بين الإضاءة والمادة"""

        # حساب ثقة الفصل
        separation_confidence = (material.confidence + 0.7) / 2  # متوسط مع baseline

        # تحديد الهيمنة
        is_lighting_dominant = lighting.intensity > 0.7 or lighting.uniformity < 0.3
        is_material_dominant = material.reflectivity < 0.4 and material.texture_roughness > 0.5

        # توليد تفسير
        explanation = self._generate_explanation(material, lighting, is_lighting_dominant, is_material_dominant)

        return LightingMaterialSeparation(
            material=material,
            lighting=lighting,
            separation_confidence=separation_confidence,
            is_lighting_dominant=is_lighting_dominant,
            is_material_dominant=is_material_dominant,
            explanation=explanation
        )

    def _generate_explanation(self,
                             material: MaterialProperties,
                             lighting: LightingAnalysis,
                             lighting_dom: bool,
                             material_dom: bool) -> str:
        """توليد تفسير للتحليل"""
        parts = []

        # Material part
        parts.append(f"Material appears to be {material.material_type.value}")
        parts.append(f"with {material.surface_property.value} surface")

        # Lighting part
        parts.append(f"Lighting shows {lighting.dominant_effect.value}")
        parts.append(f"with {lighting.color_temperature} color temperature")

        # Dominance
        if lighting_dom:
            parts.append("Changes likely due to LIGHTING variation")
        elif material_dom:
            parts.append("Changes likely due to MATERIAL properties")
        else:
            parts.append("Both lighting and material contribute to appearance")

        return ". ".join(parts)

    def _fallback_material(self) -> MaterialProperties:
        """مادة احتياطية"""
        return MaterialProperties(
            material_type=MaterialType.UNKNOWN,
            surface_property=SurfaceProperty.MATTE,
            reflectivity=0.5,
            texture_roughness=0.5,
            color_uniformity=0.5,
            confidence=0.1
        )

    def _fallback_lighting(self) -> LightingAnalysis:
        """إضاءة احتياطية"""
        return LightingAnalysis(
            dominant_effect=LightingEffect.AMBIENT,
            intensity=0.5,
            direction_hint=None,
            color_temperature="neutral",
            uniformity=0.5
        )

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات المحلل"""
        return {
            "total_analyzed": self.total_analyzed,
            "material_type_distribution": {
                mat_type.value: count
                for mat_type, count in self.material_type_counts.items()
            },
            "image_analyzer_available": IMAGE_ANALYZER_AVAILABLE
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🔬 Material Analyzer - Test")
    print("=" * 70)
    print()

    analyzer = MaterialAnalyzer()

    print(f"ImageAnalyzer Available: {analyzer.image_analyzer is not None}")
    print()

    # في الاستخدام الحقيقي:
    # result = analyzer.analyze("path/to/image.jpg")
    # print(f"Material: {result.material.material_type.value}")
    # print(f"Reflectivity: {result.material.reflectivity:.2f}")
    # print(f"Lighting Effect: {result.lighting.dominant_effect.value}")
    # print(f"Explanation: {result.explanation}")

    print("=" * 70)
    print("✅ Material Analyzer initialized!")
    print("   Capabilities:")
    print("   - Material type classification (metal/wood/plastic/fabric/...)")
    print("   - Surface property analysis (glossy/matte/rough/smooth)")
    print("   - Reflectivity calculation")
    print("   - Texture roughness analysis")
    print("   - Lighting effect detection")
    print("   - Lighting vs Material separation")
    print()
