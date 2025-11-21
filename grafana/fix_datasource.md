# 🔧 إصلاح مشكلة "No Data" في Grafana Dashboard

## المشكلة
Dashboard يظهر "No data" في جميع الـ Panels مع علامة تحذير حمراء ⚠️

## السبب
الـ Data Source UID في Dashboard (المسمى `noogh_api`) لا يتطابق مع الـ UID الفعلي لـ Data Source في Grafana.

---

## ✅ الحل 1: إعادة إنشاء Data Source بالـ UID الصحيح (الأسرع)

### الخطوة 1: احذف Data Source القديم

1. اذهب إلى **⚙️ Administration** → **Data sources**
2. ابحث عن "Noogh System API"
3. انقر عليه
4. في الأسفل، انقر **Delete** (باللون الأحمر)
5. تأكيد الحذف

### الخطوة 2: أنشئ Data Source جديد بالـ UID المطلوب

**⚠️ للأسف**: Grafana لا يسمح بتحديد UID يدوياً من الواجهة.

**الحل البديل**: استخدم Grafana API:

```bash
curl -X POST http://localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{
    "uid": "noogh_api",
    "name": "Noogh System API",
    "type": "yesoreyeram-infinity-datasource",
    "url": "http://localhost:8000",
    "access": "proxy",
    "isDefault": false
  }'
```

**ملاحظة**: غير `admin:admin` إلى username:password الخاص بـ Grafana.

---

## ✅ الحل 2: تعديل Dashboard JSON (الطريقة اليدوية)

### إذا كان Data Source UID الحالي مثلاً: `df4ga7s13zfgge`

1. **احصل على UID الحالي:**
   - اذهب إلى **Data sources** → **Noogh System API**
   - من الـ URL: `http://localhost:3000/datasources/edit/df4ga7s13zfgge`
   - انسخ: `df4ga7s13zfgge`

2. **عدّل Dashboard JSON:**
   ```bash
   cd /home/noogh/projects/noogh_unified_system/grafana

   # عمل نسخة احتياطية
   cp noogh_unified_system_dashboard.json noogh_unified_system_dashboard.json.backup

   # استبدال noogh_api بالـ UID الصحيح (غيّر df4ga7s13zfgge إلى UID الخاص بك)
   sed -i 's/"uid": "noogh_api"/"uid": "df4ga7s13zfgge"/g' noogh_unified_system_dashboard.json
   ```

3. **احذف Dashboard القديم وأعد الاستيراد:**
   - في Grafana: اذهب إلى Dashboard
   - **Settings** (⚙️ أعلى اليمين) → **Delete dashboard**
   - أعد استيراد الملف المعدّل

---

## ✅ الحل 3: تعديل Data Source في Dashboard مباشرة (الأبسط)

### في Grafana Dashboard:

1. افتح Dashboard المستورد
2. انقر **Edit** أعلى اليمين
3. اختر أي Panel واضغط عليها
4. في الأسفل تحت **Query**، ستجد:
   - **Data source**: يظهر خطأ أو "not found"
5. من القائمة المنسدلة، اختر **"Noogh System API"** الصحيح
6. انقر **Apply**
7. **كرر هذا لكل Panel** (24 panel!) ❌ هذا صعب!

**OR** الطريقة السريعة:

1. انقر **Dashboard settings** (⚙️ أعلى اليمين)
2. اذهب إلى **JSON Model**
3. ابحث عن `"uid": "noogh_api"` (Ctrl+F)
4. استبدل الكل بـ UID الصحيح (استخدم Find & Replace)
5. **Save JSON** ثم **Save dashboard**

---

## 🧪 التحقق من النجاح

بعد تطبيق أي حل أعلاه:

1. رجّع Dashboard (Refresh)
2. يجب أن تختفي علامات التحذير الحمراء ⚠️
3. يجب أن تظهر البيانات:
   - System Health: **80%**
   - Active Components: **4**
   - Ministers Table: **14 ministers**
   - Knowledge Progress: **89%**

---

## 📞 إذا استمرت المشكلة

### تحقق من:

1. **API Server يعمل:**
   ```bash
   curl http://localhost:8000/api/system/metrics | jq '. | length'
   # يجب أن يرجع: 23
   ```

2. **Data Source متصل:**
   - اذهب إلى Data sources → Noogh System API
   - انقر **Save & test**
   - يجب أن يظهر: ✅ **Health check successful**

3. **Panel Query صحيح:**
   - افتح أي Panel في Edit mode
   - تحت Query، تحقق من:
     - **Type**: JSON
     - **Parser**: Backend
     - **Source**: URL
     - **URL**: `/api/system/metrics` (أو المسار الصحيح)
   - انقر **Run Query**
   - يجب أن تظهر البيانات في الأسفل

---

## 🚀 التوصية

**استخدم الحل 2** (تعديل Dashboard JSON عبر Find & Replace):

1. احصل على UID الحالي من Data Source
2. افتح Dashboard settings → JSON Model
3. Find & Replace: `"noogh_api"` → `"your_actual_uid"`
4. Save
5. Refresh

**هذا سيصلح جميع الـ 24 Panels دفعة واحدة!** ⚡
