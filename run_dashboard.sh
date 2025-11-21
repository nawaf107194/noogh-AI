#!/bin/bash
# سكريبت تشغيل نظام نوغ - نسخة مبسطة

echo "🚀 تشغيل نظام نوغ..."
echo ""

# الانتقال للمجلد الرئيسي
cd /home/noogh/projects/noogh_unified_system

# تفعيل البيئة الافتراضية
echo "⚙️  تفعيل البيئة الافتراضية..."
source venv/bin/activate

# إنشاء مجلد السجلات
mkdir -p logs

# التشغيل
echo ""
echo "▶️  تشغيل لوحة التحكم..."
echo "   📍 افتح المتصفح على: http://localhost:8501"
echo ""

# تشغيل Dashboard فقط (للتسهيل)
streamlit run src/interface/dashboard.py \
  --server.port 8501 \
  --theme.base "dark" \
  --theme.primaryColor "#2962FF"
