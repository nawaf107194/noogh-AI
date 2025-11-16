"""
Image Analyzer - محلل الصور
=============================

يحلل الصور ويستخرج معلومات أساسية (الألوان، الأشكال، التكوين، إلخ)

Addresses Computer Vision capabilities from Self-Audit

Author: Noogh AI Team
Date: 2025-11-10
Priority: HIGH
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import os
import hashlib


class ImageType(str, Enum):
    """نوع الصورة"""
    PHOTO = "photo"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    DOCUMENT = "document"
    ARTWORK = "artwork"
    UNKNOWN = "unknown"


class ColorScheme(str, Enum):
    """نظام الألوان"""
    GRAYSCALE = "grayscale"
    MONOCHROME = "monochrome"
    COLORFUL = "colorful"
    PASTEL = "pastel"
    VIBRANT = "vibrant"
    DARK = "dark"
    LIGHT = "light"


@dataclass
class ColorInfo:
    """معلومات الألوان"""
    dominant_colors: List[Tuple[int, int, int]]  # RGB tuples
    color_scheme: ColorScheme
    color_diversity: float  # 0.0-1.0
    brightness_avg: float  # 0-255


@dataclass
class DimensionInfo:
    """معلومات الأبعاد"""
    width: int
    height: int
    aspect_ratio: float
    megapixels: float
    orientation: str  # "landscape", "portrait", "square"


@dataclass
class ContentAnalysis:
    """تحليل المحتوى"""
    has_text: bool
    text_density: float  # 0.0-1.0
    estimated_complexity: float  # 0.0-1.0 (بسيط → معقد)
    edges_detected: int
    regions_count: int


@dataclass
class ImageAnalysisResult:
    """نتيجة تحليل الصورة"""
    image_path: str
    file_size_bytes: int
    image_type: ImageType
    dimensions: DimensionInfo
    colors: ColorInfo
    content: ContentAnalysis
    file_hash: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "file_size_mb": self.file_size_bytes / (1024 * 1024),
            "image_type": self.image_type.value,
            "dimensions": {
                "width": self.dimensions.width,
                "height": self.dimensions.height,
                "aspect_ratio": self.dimensions.aspect_ratio,
                "megapixels": self.dimensions.megapixels,
                "orientation": self.dimensions.orientation
            },
            "colors": {
                "dominant_colors": self.colors.dominant_colors,
                "color_scheme": self.colors.color_scheme.value,
                "color_diversity": self.colors.color_diversity,
                "brightness": self.colors.brightness_avg
            },
            "content": {
                "has_text": self.content.has_text,
                "text_density": self.content.text_density,
                "complexity": self.content.estimated_complexity,
                "edges": self.content.edges_detected,
                "regions": self.content.regions_count
            },
            "file_hash": self.file_hash,
            "timestamp": self.timestamp.isoformat()
        }


class ImageAnalyzer:
    """
    محلل الصور

    يحلل:
    1. أبعاد ونوع الصورة
    2. الألوان السائدة والنظام اللوني
    3. محتوى الصورة (نصوص، تعقيد، إلخ)
    4. معلومات الملف
    """

    def __init__(self):
        # محاولة تحميل PIL/Pillow
        self.pil_available = self._check_pil()

        # محاولة تحميل OpenCV
        self.opencv_available = self._check_opencv()

        # إحصائيات
        self.total_analyzed = 0
        self.analysis_history: List[ImageAnalysisResult] = []

    def _check_pil(self) -> bool:
        """فحص توفر PIL/Pillow"""
        try:
            from PIL import Image
            return True
        except Exception as e:
            return False

    def _check_opencv(self) -> bool:
        """فحص توفر OpenCV"""
        try:
            import cv2
            return True
        except Exception as e:
            return False

    def analyze_image(self, image_path: str) -> ImageAnalysisResult:
        """
        تحليل صورة كامل

        Args:
            image_path: مسار الصورة

        Returns:
            ImageAnalysisResult
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # معلومات الملف
        file_size = os.path.getsize(image_path)
        file_hash = self._compute_hash(image_path)

        if self.pil_available:
            result = self._analyze_with_pil(image_path, file_size, file_hash)
        else:
            result = self._analyze_basic(image_path, file_size, file_hash)

        self.total_analyzed += 1
        self.analysis_history.append(result)

        return result

    def _analyze_with_pil(self,
                         image_path: str,
                         file_size: int,
                         file_hash: str) -> ImageAnalysisResult:
        """تحليل باستخدام PIL/Pillow"""
        from PIL import Image
        import numpy as np

        # فتح الصورة
        image = Image.open(image_path)

        # معلومات الأبعاد
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0
        megapixels = (width * height) / 1_000_000

        if aspect_ratio > 1.2:
            orientation = "landscape"
        elif aspect_ratio < 0.8:
            orientation = "portrait"
        else:
            orientation = "square"

        dimensions = DimensionInfo(
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            orientation=orientation
        )

        # تحليل الألوان
        colors = self._analyze_colors_pil(image)

        # تحليل المحتوى
        content = self._analyze_content_pil(image)

        # تحديد نوع الصورة
        image_type = self._determine_image_type(colors, content)

        return ImageAnalysisResult(
            image_path=image_path,
            file_size_bytes=file_size,
            image_type=image_type,
            dimensions=dimensions,
            colors=colors,
            content=content,
            file_hash=file_hash,
            timestamp=datetime.now(timezone.utc)
        )

    def _analyze_colors_pil(self, image) -> ColorInfo:
        """تحليل الألوان باستخدام PIL"""
        from PIL import Image
        import numpy as np

        # تحويل لـ RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # تصغير الصورة لتسريع التحليل
        small_image = image.resize((100, 100))
        pixels = np.array(small_image)

        # حساب الألوان السائدة (تبسيط)
        # في النظام الكامل: استخدام k-means clustering
        avg_color = pixels.mean(axis=(0, 1)).astype(int)
        dominant_colors = [tuple(avg_color)]

        # حساب السطوع المتوسط
        brightness_avg = float(np.mean(pixels))

        # حساب تنوع الألوان
        color_std = float(np.std(pixels))
        color_diversity = min(color_std / 128.0, 1.0)  # نسبة من الانحراف المعياري

        # تحديد نظام الألوان
        if color_diversity < 0.1:
            color_scheme = ColorScheme.MONOCHROME
        elif brightness_avg < 85:
            color_scheme = ColorScheme.DARK
        elif brightness_avg > 170:
            color_scheme = ColorScheme.LIGHT
        elif color_diversity > 0.5:
            color_scheme = ColorScheme.VIBRANT
        else:
            color_scheme = ColorScheme.COLORFUL

        return ColorInfo(
            dominant_colors=dominant_colors,
            color_scheme=color_scheme,
            color_diversity=color_diversity,
            brightness_avg=brightness_avg
        )

    def _analyze_content_pil(self, image) -> ContentAnalysis:
        """تحليل المحتوى باستخدام PIL"""
        import numpy as np

        # تحويل لـ grayscale
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)

        # كشف الحواف (تبسيط - في النظام الكامل: استخدام Sobel/Canny)
        edges = self._simple_edge_detection(gray_array)
        edges_count = int(np.sum(edges > 0))

        # تقدير التعقيد من عدد الحواف
        total_pixels = gray_array.size
        edge_ratio = edges_count / total_pixels
        estimated_complexity = min(edge_ratio * 10, 1.0)  # نسبة تقريبية

        # كشف النصوص (تبسيط)
        # في النظام الكامل: استخدام OCR للكشف عن وجود نصوص
        variance = float(np.var(gray_array))
        has_text = variance > 1000  # نصوص عادة تعطي variance عالي

        text_density = 0.0  # placeholder - يحتاج OCR للدقة

        # عدد المناطق (تبسيط)
        regions_count = max(1, int(estimated_complexity * 10))

        return ContentAnalysis(
            has_text=has_text,
            text_density=text_density,
            estimated_complexity=estimated_complexity,
            edges_detected=edges_count,
            regions_count=regions_count
        )

    def _simple_edge_detection(self, gray_array) -> 'np.ndarray':
        """كشف حواف بسيط"""
        import numpy as np

        # Sobel filter بسيط
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # تطبيق الفلتر (تبسيط - بدون padding)
        height, width = gray_array.shape
        edges = np.zeros((height - 2, width - 2))

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = gray_array[i-1:i+2, j-1:j+2]
                gx = np.sum(region * kernel_x)
                gy = np.sum(region * kernel_y)
                edges[i-1, j-1] = np.sqrt(gx**2 + gy**2)

        # threshold
        threshold = 100
        edges = (edges > threshold).astype(np.uint8) * 255

        return edges

    def _determine_image_type(self, colors: ColorInfo, content: ContentAnalysis) -> ImageType:
        """تحديد نوع الصورة"""
        # قواعد بسيطة
        if content.has_text and content.text_density > 0.3:
            return ImageType.DOCUMENT

        if colors.color_scheme in [ColorScheme.MONOCHROME, ColorScheme.GRAYSCALE] and content.has_text:
            return ImageType.SCREENSHOT

        if content.estimated_complexity < 0.2:
            return ImageType.DIAGRAM

        if colors.color_scheme == ColorScheme.VIBRANT and content.estimated_complexity > 0.5:
            return ImageType.ARTWORK

        return ImageType.PHOTO

    def _analyze_basic(self,
                      image_path: str,
                      file_size: int,
                      file_hash: str) -> ImageAnalysisResult:
        """تحليل بسيط (fallback بدون PIL)"""
        # معلومات أساسية فقط
        return ImageAnalysisResult(
            image_path=image_path,
            file_size_bytes=file_size,
            image_type=ImageType.UNKNOWN,
            dimensions=DimensionInfo(
                width=0,
                height=0,
                aspect_ratio=1.0,
                megapixels=0.0,
                orientation="unknown"
            ),
            colors=ColorInfo(
                dominant_colors=[(128, 128, 128)],
                color_scheme=ColorScheme.COLORFUL,
                color_diversity=0.5,
                brightness_avg=128.0
            ),
            content=ContentAnalysis(
                has_text=False,
                text_density=0.0,
                estimated_complexity=0.5,
                edges_detected=0,
                regions_count=1
            ),
            file_hash=file_hash,
            timestamp=datetime.now(timezone.utc)
        )

    def _compute_hash(self, file_path: str) -> str:
        """حساب hash للملف"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # قراءة على دفعات لملفات كبيرة
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def compare_images(self, image_path1: str, image_path2: str) -> Dict[str, Any]:
        """
        مقارنة صورتين

        Returns:
            تقرير المقارنة
        """
        analysis1 = self.analyze_image(image_path1)
        analysis2 = self.analyze_image(image_path2)

        # مقارنة الهاش (تطابق تام)
        identical = analysis1.file_hash == analysis2.file_hash

        # مقارنة الأبعاد
        same_dimensions = (
            analysis1.dimensions.width == analysis2.dimensions.width and
            analysis1.dimensions.height == analysis2.dimensions.height
        )

        # مقارنة الألوان (تشابه تقريبي)
        color_similarity = 1.0 - abs(
            analysis1.colors.color_diversity - analysis2.colors.color_diversity
        )

        return {
            "identical": identical,
            "same_dimensions": same_dimensions,
            "color_similarity": color_similarity,
            "type_match": analysis1.image_type == analysis2.image_type,
            "image1": analysis1.to_dict(),
            "image2": analysis2.to_dict()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات المحلل"""
        if self.total_analyzed == 0:
            return {
                "status": "no_data",
                "pil_available": self.pil_available,
                "opencv_available": self.opencv_available
            }

        return {
            "total_analyzed": self.total_analyzed,
            "pil_available": self.pil_available,
            "opencv_available": self.opencv_available,
            "image_types": {
                image_type.value: sum(
                    1 for r in self.analysis_history
                    if r.image_type == image_type
                )
                for image_type in ImageType
            }
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🖼️  Image Analyzer - Test")
    print("=" * 70)

    analyzer = ImageAnalyzer()

    print(f"PIL available: {analyzer.pil_available}")
    print(f"OpenCV available: {analyzer.opencv_available}")
    print()

    # في الاستخدام الحقيقي:
    # result = analyzer.analyze_image("path/to/image.png")
    # print(result.to_dict())

    print("=" * 70)
    print("📊 Statistics:")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("✅ Image Analyzer initialized successfully!")
    print("   To use: result = analyzer.analyze_image('image.png')")
    print()
