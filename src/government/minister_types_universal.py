#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Government System - Minister Types
نظام الحكومة الشامل - أنواع الوزراء

Version: 4.0.0 - Universal System (Not Just Trading)
"""

from enum import Enum
from typing import Dict, List, Any


class MinisterCategory(Enum):
    """فئات الوزراء الرئيسية"""
    KNOWLEDGE = "knowledge"           # المعرفة والتعليم
    SECURITY = "security"             # الأمن والحماية
    CREATIVE = "creative"             # التطوير والإبداع
    ANALYSIS = "analysis"             # التحليل والقرار
    AI_CORE = "ai_core"              # الذكاء الاصطناعي
    COMMUNICATION = "communication"   # التواصل والإدارة
    FINANCE = "finance"              # المالية (جزء صغير من النظام)


class MinisterType(Enum):
    """
    أنواع الوزراء - نظام شامل متنوع

    النظام الجديد يشمل 7 فئات رئيسية و 17 وزير
    التداول المالي أصبح جزء صغير من النظام (4 من 17)
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📚 KNOWLEDGE SECTOR - قطاع المعرفة والتعليم
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    EDUCATION = "education"           # وزير التعليم - تعليم ذكي مخصص
    RESEARCH = "research"             # وزير البحث - البحث العلمي والابتكار
    KNOWLEDGE = "knowledge"           # وزير المعرفة - إدارة قاعدة المعرفة

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🛡️ SECURITY SECTOR - قطاع الأمن والحماية
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    SECURITY = "security"             # وزير الأمن - حماية النظام
    PRIVACY = "privacy"               # وزير الخصوصية - حماية البيانات

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 💡 CREATIVE SECTOR - قطاع التطوير والإبداع
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DEVELOPMENT = "development"       # وزير التطوير - كتابة وتحسين الكود
    CREATIVITY = "creativity"         # وزير الإبداع - الابتكار والأفكار

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📊 ANALYSIS SECTOR - قطاع التحليل والقرار
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ANALYSIS = "analysis"             # وزير التحليل - تحليل البيانات الشامل
    STRATEGY = "strategy"             # وزير الاستراتيجية - التخطيط الاستراتيجي

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🤖 AI CORE SECTOR - قطاع الذكاء الاصطناعي
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    TRAINING = "training"             # وزير التدريب - تدريب النماذج
    REASONING = "reasoning"           # وزير الاستدلال - التفكير المنطقي

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 💬 COMMUNICATION SECTOR - قطاع التواصل والإدارة
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    COMMUNICATION = "communication"   # وزير التواصل - الرد والتقارير
    RESOURCES = "resources"           # وزير الموارد - إدارة الموارد

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 💰 FINANCE SECTOR - قطاع المالية (جزء صغير من النظام)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    FINANCE = "finance"               # وزير المالية - إدارة المال والتداول
    # ملاحظة: الوزراء الماليون السابقون (Technical, Quantitative, Risk, Portfolio)
    # سيتم دمجهم كأدوات ضمن وزير التحليل أو وزير المالية


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# تصنيف الوزراء حسب الفئات
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MINISTER_CATEGORIES: Dict[MinisterCategory, List[MinisterType]] = {
    MinisterCategory.KNOWLEDGE: [
        MinisterType.EDUCATION,
        MinisterType.RESEARCH,
        MinisterType.KNOWLEDGE,
    ],
    MinisterCategory.SECURITY: [
        MinisterType.SECURITY,
        MinisterType.PRIVACY,
    ],
    MinisterCategory.CREATIVE: [
        MinisterType.DEVELOPMENT,
        MinisterType.CREATIVITY,
    ],
    MinisterCategory.ANALYSIS: [
        MinisterType.ANALYSIS,
        MinisterType.STRATEGY,
    ],
    MinisterCategory.AI_CORE: [
        MinisterType.TRAINING,
        MinisterType.REASONING,
    ],
    MinisterCategory.COMMUNICATION: [
        MinisterType.COMMUNICATION,
        MinisterType.RESOURCES,
    ],
    MinisterCategory.FINANCE: [
        MinisterType.FINANCE,
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# معلومات الوزراء
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MINISTER_INFO: Dict[MinisterType, Dict[str, str]] = {
    # Knowledge Sector
    MinisterType.EDUCATION: {
        'arabic': 'وزير التعليم',
        'english': 'Minister of Education',
        'description': 'تعليم ذكي مخصص، إنشاء دورات، تقييم المعرفة',
        'keywords': 'علمني، تعلم، دورة، شرح، مفهوم، درس',
        'gpu_enabled': True,
        'category': 'knowledge'
    },
    MinisterType.RESEARCH: {
        'arabic': 'وزير البحث والتطوير',
        'english': 'Minister of Research',
        'description': 'البحث العلمي، تحليل الأوراق، اقتراح أفكار',
        'keywords': 'ابحث، بحث، أبحاث، ورقة علمية، دراسة',
        'gpu_enabled': True,
        'category': 'knowledge'
    },
    MinisterType.KNOWLEDGE: {
        'arabic': 'وزير المعرفة',
        'english': 'Minister of Knowledge',
        'description': 'إدارة قاعدة المعرفة، Knowledge Graph، ربط المفاهيم',
        'keywords': 'احفظ، استرجع، معرفة، علاقة، ربط',
        'gpu_enabled': True,
        'category': 'knowledge'
    },

    # Security Sector
    MinisterType.SECURITY: {
        'arabic': 'وزير الأمن',
        'english': 'Minister of Security',
        'description': 'حماية النظام، كشف التهديدات، مراقبة الأمن',
        'keywords': 'أمن، حماية، تهديد، اختراق، مراقبة',
        'gpu_enabled': True,
        'category': 'security'
    },
    MinisterType.PRIVACY: {
        'arabic': 'وزير الخصوصية',
        'english': 'Minister of Privacy',
        'description': 'حماية البيانات الحساسة، تشفير، إدارة الصلاحيات',
        'keywords': 'خصوصية، تشفير، بيانات حساسة، صلاحيات',
        'gpu_enabled': True,
        'category': 'security'
    },

    # Creative Sector
    MinisterType.DEVELOPMENT: {
        'arabic': 'وزير التطوير',
        'english': 'Minister of Development',
        'description': 'كتابة الكود، تحسين الأداء، إصلاح الأخطاء',
        'keywords': 'طور، حسّن، اصلح، كود، برمجة، bug',
        'gpu_enabled': False,
        'category': 'creative'
    },
    MinisterType.CREATIVITY: {
        'arabic': 'وزير الإبداع',
        'english': 'Minister of Creativity',
        'description': 'توليد أفكار، إبداع، ابتكار، عصف ذهني',
        'keywords': 'ابتكر، فكرة، إبداع، عصف ذهني، تصميم',
        'gpu_enabled': True,
        'category': 'creative'
    },

    # Analysis Sector
    MinisterType.ANALYSIS: {
        'arabic': 'وزير التحليل',
        'english': 'Minister of Analysis',
        'description': 'تحليل البيانات الشامل، كشف الأنماط، التنبؤ',
        'keywords': 'حلل، تحليل، بيانات، نمط، اتجاه، pattern',
        'gpu_enabled': True,
        'category': 'analysis'
    },
    MinisterType.STRATEGY: {
        'arabic': 'وزير الاستراتيجية',
        'english': 'Minister of Strategy',
        'description': 'التخطيط الاستراتيجي، تحليل SWOT، تحديد الأولويات',
        'keywords': 'خطة، استراتيجية، swot، أولويات، تخطيط',
        'gpu_enabled': True,
        'category': 'analysis'
    },

    # AI Core Sector
    MinisterType.TRAINING: {
        'arabic': 'وزير التدريب',
        'english': 'Minister of Training',
        'description': 'تدريب النماذج، Fine-tuning، تقييم النماذج',
        'keywords': 'درّب، تدريب، نموذج، model، fine-tune',
        'gpu_enabled': True,
        'category': 'ai_core'
    },
    MinisterType.REASONING: {
        'arabic': 'وزير الاستدلال',
        'english': 'Minister of Reasoning',
        'description': 'التفكير المنطقي، حل المشاكل المعقدة، الاستدلال',
        'keywords': 'حل، مشكلة، فكر، استدلال، منطق، reasoning',
        'gpu_enabled': True,
        'category': 'ai_core'
    },

    # Communication Sector
    MinisterType.COMMUNICATION: {
        'arabic': 'وزير التواصل',
        'english': 'Minister of Communication',
        'description': 'الرد على الاستفسارات، كتابة التقارير، الترجمة',
        'keywords': 'اكتب، تقرير، ترجم، لخص، رد',
        'gpu_enabled': False,
        'category': 'communication'
    },
    MinisterType.RESOURCES: {
        'arabic': 'وزير الموارد',
        'english': 'Minister of Resources',
        'description': 'إدارة موارد النظام، مراقبة GPU/CPU، جدولة المهام',
        'keywords': 'موارد، gpu، cpu، memory، جدولة',
        'gpu_enabled': False,
        'category': 'communication'
    },

    # Finance Sector (Small Part)
    MinisterType.FINANCE: {
        'arabic': 'وزير المالية',
        'english': 'Minister of Finance',
        'description': 'التداول المالي، إدارة المحفظة، تحليل الأسواق (جزء من النظام)',
        'keywords': 'تداول، شراء، بيع، سهم، محفظة، سوق',
        'gpu_enabled': True,
        'category': 'finance'
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task Types - أنواع المهام
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TaskCategory(Enum):
    """فئات المهام"""
    EDUCATIONAL = "educational"       # تعليمية
    RESEARCH = "research"            # بحثية
    SECURITY = "security"            # أمنية
    DEVELOPMENT = "development"      # تطويرية
    CREATIVE = "creative"            # إبداعية
    ANALYTICAL = "analytical"        # تحليلية
    STRATEGIC = "strategic"          # استراتيجية
    AI_TRAINING = "ai_training"      # تدريب AI
    COMMUNICATION = "communication"  # تواصلية
    FINANCIAL = "financial"          # مالية (جزء صغير)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_minister_by_keywords(text: str) -> List[MinisterType]:
    """تحديد الوزراء المناسبين بناءً على الكلمات المفتاحية"""
    text_lower = text.lower()
    matching_ministers = []

    for minister_type, info in MINISTER_INFO.items():
        keywords = info['keywords'].lower().split('، ')
        if any(keyword in text_lower for keyword in keywords):
            matching_ministers.append(minister_type)

    return matching_ministers


def get_minister_category(minister_type: MinisterType) -> MinisterCategory:
    """الحصول على فئة الوزير"""
    for category, ministers in MINISTER_CATEGORIES.items():
        if minister_type in ministers:
            return category
    return MinisterCategory.COMMUNICATION  # Default


def is_gpu_enabled(minister_type: MinisterType) -> bool:
    """هل الوزير يستخدم GPU؟"""
    info = MINISTER_INFO.get(minister_type, {})
    return info.get('gpu_enabled', False)


def get_minister_arabic_name(minister_type: MinisterType) -> str:
    """الحصول على الاسم العربي للوزير"""
    info = MINISTER_INFO.get(minister_type, {})
    return info.get('arabic', minister_type.value)


def get_all_ministers_by_category() -> Dict[str, List[str]]:
    """عرض جميع الوزراء مصنفين حسب الفئات"""
    result = {}
    for category, ministers in MINISTER_CATEGORIES.items():
        result[category.value] = [
            get_minister_arabic_name(m) for m in ministers
        ]
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_system_statistics() -> Dict[str, Any]:
    """إحصائيات النظام"""
    total_ministers = len(MinisterType)
    gpu_enabled = sum(1 for m in MinisterType if is_gpu_enabled(m))

    category_counts = {
        category.value: len(ministers)
        for category, ministers in MINISTER_CATEGORIES.items()
    }

    return {
        'total_ministers': total_ministers,
        'gpu_enabled_ministers': gpu_enabled,
        'cpu_only_ministers': total_ministers - gpu_enabled,
        'categories': category_counts,
        'finance_ministers': category_counts.get('finance', 0),
        'non_finance_ministers': total_ministers - category_counts.get('finance', 0),
        'finance_percentage': f"{(category_counts.get('finance', 0) / total_ministers * 100):.1f}%"
    }


if __name__ == "__main__":
    print("="*70)
    print("🏛️ UNIVERSAL GOVERNMENT SYSTEM - MINISTER TYPES")
    print("="*70)

    stats = get_system_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Total Ministers: {stats['total_ministers']}")
    print(f"   GPU-Enabled: {stats['gpu_enabled_ministers']}")
    print(f"   CPU-Only: {stats['cpu_only_ministers']}")
    print(f"   Finance Ministers: {stats['finance_ministers']} ({stats['finance_percentage']})")
    print(f"   Non-Finance Ministers: {stats['non_finance_ministers']}")

    print(f"\n👥 Ministers by Category:")
    ministers_by_cat = get_all_ministers_by_category()
    for category, ministers in ministers_by_cat.items():
        print(f"\n   {category.upper()}:")
        for minister in ministers:
            print(f"      • {minister}")

    print("\n" + "="*70)
    print("✅ Universal system with diverse ministers!")
    print("   Financial trading is now just ~6% of the system")
    print("="*70)
