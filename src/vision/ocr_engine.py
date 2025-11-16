"""
OCR Engine - محرك قراءة النصوص من الصور
==========================================

يستخرج النصوص من الصور بعدة لغات (عربي، إنجليزي، إلخ)

Addresses Q6 from Self-Audit (Computer Vision)

Author: Noogh AI Team
Date: 2025-11-10
Priority: HIGH
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import os
import re
from pathlib import Path


class OCRLanguage(str, Enum):
    """اللغات المدعومة"""
    ARABIC = "ara"
    ENGLISH = "eng"
    FRENCH = "fra"
    SPANISH = "spa"
    AUTO = "auto"


class OCRConfidence(str, Enum):
    """مستوى الثقة في النتيجة"""
    HIGH = "high"  # > 90%
    MEDIUM = "medium"  # 70-90%
    LOW = "low"  # 50-70%
    VERY_LOW = "very_low"  # < 50%


@dataclass
class TextRegion:
    """منطقة نص في الصورة"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # 0.0-1.0
    language: Optional[str] = None


@dataclass
class OCRResult:
    """نتيجة OCR"""
    full_text: str
    regions: List[TextRegion]
    detected_languages: List[str]
    average_confidence: float
    confidence_level: OCRConfidence
    processing_time_ms: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_text": self.full_text,
            "regions": [
                {
                    "text": r.text,
                    "bbox": r.bbox,
                    "confidence": r.confidence,
                    "language": r.language
                }
                for r in self.regions
            ],
            "detected_languages": self.detected_languages,
            "average_confidence": self.average_confidence,
            "confidence_level": self.confidence_level.value,
            "processing_time_ms": self.processing_time_ms
        }


class OCREngine:
    """
    محرك قراءة النصوص من الصور

    يدعم:
    1. قراءة النصوص العربية والإنجليزية
    2. كشف اللغة تلقائياً
    3. استخراج مناطق النصوص
    4. قياس الثقة في النتائج
    """

    def __init__(self, default_language: OCRLanguage = OCRLanguage.AUTO):
        """
        Args:
            default_language: اللغة الافتراضية
        """
        self.default_language = default_language
        self.supported_languages = [
            OCRLanguage.ARABIC,
            OCRLanguage.ENGLISH,
            OCRLanguage.FRENCH,
            OCRLanguage.SPANISH
        ]

        # محاولة تحميل tesseract إن وجد
        self.tesseract_available = self._check_tesseract()

        # إحصائيات
        self.total_processed = 0
        self.total_errors = 0
        self.processing_history: List[OCRResult] = []

    def _check_tesseract(self) -> bool:
        """فحص توفر tesseract OCR"""
        try:
            import pytesseract
            # محاولة الوصول لـ tesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            # Error caught: {e}
            return False

    def extract_text(self,
                    image_path: str,
                    language: OCRLanguage = None,
                    detect_regions: bool = True) -> OCRResult:
        """
        استخراج النصوص من صورة

        Args:
            image_path: مسار الصورة
            language: اللغة (None = استخدام الافتراضي)
            detect_regions: كشف مناطق النصوص المنفصلة

        Returns:
            OCRResult
        """
        start_time = datetime.now(timezone.utc)

        if language is None:
            language = self.default_language

        # فحص وجود الملف
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            if self.tesseract_available:
                result = self._extract_with_tesseract(image_path, language, detect_regions)
            else:
                # Fallback: استخراج بسيط (للتطوير/الاختبار)
                result = self._extract_basic(image_path, language)

            # حساب وقت المعالجة
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result.processing_time_ms = processing_time
            result.timestamp = start_time

            self.total_processed += 1
            self.processing_history.append(result)

            return result

        except Exception as e:
            self.total_errors += 1
            raise RuntimeError(f"OCR extraction failed: {str(e)}")

    def _extract_with_tesseract(self,
                                image_path: str,
                                language: OCRLanguage,
                                detect_regions: bool) -> OCRResult:
        """استخراج باستخدام Tesseract OCR"""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("pytesseract or PIL not installed. Install with: pip install pytesseract pillow")

        # فتح الصورة
        image = Image.open(image_path)

        # تحديد اللغة
        if language == OCRLanguage.AUTO:
            # محاولة الكشف التلقائي
            lang_code = "ara+eng"  # دعم عربي وإنجليزي معاً
        else:
            lang_code = language.value

        # استخراج النص الكامل
        full_text = pytesseract.image_to_string(image, lang=lang_code)

        # استخراج المناطق إن طُلب
        regions = []
        detected_languages = set()

        if detect_regions:
            # استخراج بيانات تفصيلية
            data = pytesseract.image_to_data(image, lang=lang_code, output_type=pytesseract.Output.DICT)

            # تجميع المناطق
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:  # تجاهل النصوص الفارغة
                    conf = float(data['conf'][i]) / 100.0  # تحويل لنطاق 0-1

                    # تحديد اللغة
                    detected_lang = self._detect_language_simple(text)
                    detected_languages.add(detected_lang)

                    region = TextRegion(
                        text=text,
                        bbox=(
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        ),
                        confidence=conf,
                        language=detected_lang
                    )
                    regions.append(region)

        # حساب الثقة المتوسطة
        if regions:
            avg_confidence = sum(r.confidence for r in regions) / len(regions)
        else:
            # fallback: نص واحد مع ثقة متوسطة
            avg_confidence = 0.7
            detected_languages.add("unknown")

        confidence_level = self._classify_confidence(avg_confidence)

        return OCRResult(
            full_text=full_text.strip(),
            regions=regions,
            detected_languages=list(detected_languages),
            average_confidence=avg_confidence,
            confidence_level=confidence_level,
            processing_time_ms=0.0,  # سيتم تحديثها
            timestamp=datetime.now(timezone.utc)
        )

    def _extract_basic(self, image_path: str, language: OCRLanguage) -> OCRResult:
        """
        استخراج بسيط (للحالات التي لا يتوفر فيها tesseract)

        ملحوظة: هذا placeholder للتطوير - في الإنتاج يُفضل استخدام tesseract
        """
        # محاكاة نتيجة OCR بسيطة
        dummy_text = f"[OCR Placeholder: Image at {image_path}]"

        return OCRResult(
            full_text=dummy_text,
            regions=[
                TextRegion(
                    text=dummy_text,
                    bbox=(0, 0, 100, 50),
                    confidence=0.5,
                    language=language.value if language != OCRLanguage.AUTO else "eng"
                )
            ],
            detected_languages=["eng"],
            average_confidence=0.5,
            confidence_level=OCRConfidence.LOW,
            processing_time_ms=0.0,
            timestamp=datetime.now(timezone.utc)
        )

    def _detect_language_simple(self, text: str) -> str:
        """كشف اللغة بشكل بسيط"""
        # فحص الأحرف العربية
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        if len(arabic_chars) > len(text) * 0.3:  # أكثر من 30% أحرف عربية
            return "ara"

        # فحص الأحرف اللاتينية
        latin_chars = re.findall(r'[a-zA-Z]', text)
        if len(latin_chars) > len(text) * 0.5:
            return "eng"

        return "unknown"

    def _classify_confidence(self, confidence: float) -> OCRConfidence:
        """تصنيف مستوى الثقة"""
        if confidence >= 0.9:
            return OCRConfidence.HIGH
        elif confidence >= 0.7:
            return OCRConfidence.MEDIUM
        elif confidence >= 0.5:
            return OCRConfidence.LOW
        else:
            return OCRConfidence.VERY_LOW

    def extract_text_from_pdf(self, pdf_path: str, page_number: int = 0) -> OCRResult:
        """
        استخراج النصوص من صفحة PDF

        Args:
            pdf_path: مسار ملف PDF
            page_number: رقم الصفحة (0 = الأولى)
        """
        try:
            from pdf2image import convert_from_path
            import tempfile
        except ImportError:
            raise ImportError("pdf2image not installed. Install with: pip install pdf2image")

        # تحويل صفحة PDF لصورة
        with tempfile.TemporaryDirectory() as tmp_dir:
            images = convert_from_path(
                pdf_path,
                first_page=page_number + 1,
                last_page=page_number + 1,
                output_folder=tmp_dir
            )

            if not images:
                raise ValueError("Could not convert PDF page to image")

            # حفظ الصورة مؤقتاً
            temp_image_path = os.path.join(tmp_dir, "temp_page.png")
            images[0].save(temp_image_path)

            # استخراج النص
            return self.extract_text(temp_image_path)

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات المحرك"""
        if self.total_processed == 0:
            return {
                "status": "no_data",
                "tesseract_available": self.tesseract_available
            }

        avg_confidence = sum(r.average_confidence for r in self.processing_history) / len(self.processing_history)
        avg_time = sum(r.processing_time_ms for r in self.processing_history) / len(self.processing_history)

        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "success_rate": ((self.total_processed - self.total_errors) / self.total_processed) * 100,
            "average_confidence": avg_confidence,
            "average_processing_time_ms": avg_time,
            "tesseract_available": self.tesseract_available,
            "supported_languages": [lang.value for lang in self.supported_languages]
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("📷 OCR Engine - Test")
    print("=" * 70)

    engine = OCREngine(default_language=OCRLanguage.AUTO)

    print(f"Tesseract available: {engine.tesseract_available}")
    print(f"Supported languages: {[lang.value for lang in engine.supported_languages]}")
    print()

    # محاكاة استخراج نص (بدون صورة فعلية)
    print("📝 Simulating OCR extraction...")

    # في الاستخدام الحقيقي:
    # result = engine.extract_text("path/to/image.png", language=OCRLanguage.ARABIC)

    print("\n" + "=" * 70)
    print("📊 Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("✅ OCR Engine initialized successfully!")
    print("   To use: result = engine.extract_text('image.png')")
    print()
