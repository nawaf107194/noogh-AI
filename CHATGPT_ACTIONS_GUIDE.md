# دليل ربط نوغ مع ChatGPT عبر Actions
# Guide: Connect Noogh to ChatGPT via Actions

<div dir="rtl">

## ملاحظة مهمة / Important Note

**MCP (Model Context Protocol) مخصص فقط لـ Claude Desktop ولا يعمل مع ChatGPT**

MCP is only for Claude Desktop and doesn't work with ChatGPT.

**الحل البديل: استخدام ChatGPT Actions**

Alternative Solution: Use ChatGPT Actions

---

## الخطوات / Steps

### 1. إنشاء Custom GPT جديد / Create a New Custom GPT

1. اذهب إلى ChatGPT على الرابط: https://chat.openai.com/
2. اضغط على "Explore GPTs"
3. اضغط على "Create" في أعلى اليمين
4. اختر "Configure"

### 2. الإعدادات الأساسية / Basic Configuration

في صفحة Configure، املأ الحقول التالية:

**Name (الاسم):**
```
Noogh AI Assistant
```

**Description (الوصف):**
```
نظام نوغ الموحد للذكاء الاصطناعي - نظام حكومي متكامل مع 14 وزير متخصص
Noogh Unified AI System - Complete government system with 14 specialized ministers
```

**Instructions (التعليمات):**
```
You are Noogh AI, an advanced unified AI system with a complete government structure.

You have access to the following ministers through the API:
1. وزير المالية - Finance Minister
2. وزير التطوير - Development Minister
3. وزير التعليم - Education Minister
4. وزير الأمن السيبراني - Cybersecurity Minister
5. وزير الصحة - Health Minister
6. وزير التواصل - Communication Minister
7. وزير التجارة - Trading Minister
8. وزير الموارد البشرية - HR Minister
9. وزير البحث والتطوير - R&D Minister
10. وزير العمليات - Operations Minister
11. وزير الجودة - Quality Minister
12. وزير الاستراتيجية - Strategy Minister
13. وزير العلاقات - Relations Minister
14. وزير الإنتاج - Production Minister

When a user asks a question:
1. First, check system health using getGovernmentStatus
2. Identify which minister(s) are relevant to the question
3. Provide comprehensive answers using the unified brain system
4. Always respond in Arabic first, then English if requested
```

**Conversation Starters (بدايات المحادثة):**
```
ما هي حالة النظام الحكومي؟
من هم الوزراء المتاحون؟
أخبرني عن وزير التجارة
تحدث معي بالعربية
```

### 3. إضافة Actions / Add Actions

1. في صفحة Configure، انزل إلى قسم "Actions"
2. اضغط على "Create new action"
3. في حقل "Schema"، انسخ والصق المحتوى التالي:

```yaml
openapi: 3.0.0
info:
  title: Noogh Unified AI System API
  description: نظام نوغ الموحد للذكاء الاصطناعي - API للوصول إلى الوزراء والنظام الحكومي
  version: 1.0.0
servers:
  - url: https://nooogh.com
    description: Noogh Production Server

paths:
  /health:
    get:
      summary: فحص صحة النظام
      description: التحقق من أن النظام يعمل بشكل صحيح
      operationId: checkHealth
      responses:
        '200':
          description: النظام يعمل بنجاح
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  system:
                    type: string
                  version:
                    type: string

  /government/status:
    get:
      summary: حالة النظام الحكومي
      description: الحصول على معلومات عن حالة الوزراء والنظام الحكومي
      operationId: getGovernmentStatus
      responses:
        '200':
          description: معلومات النظام الحكومي
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  president:
                    type: string
                  ministers:
                    type: object

  /government/ministers:
    get:
      summary: قائمة الوزراء
      description: الحصول على قائمة جميع الوزراء في النظام (14 وزير)
      operationId: listMinisters
      responses:
        '200':
          description: قائمة الوزراء
          content:
            application/json:
              schema:
                type: object
                properties:
                  total:
                    type: integer
                  ministers:
                    type: array
                    items:
                      type: object
                      properties:
                        name:
                          type: string
                        sector:
                          type: string
                        icon:
                          type: string

  /chat:
    post:
      summary: الدردشة مع نوغ
      description: إرسال رسالة إلى نظام نوغ والحصول على رد
      operationId: chatWithNoogh
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  description: الرسالة المراد إرسالها
              required:
                - message
      responses:
        '200':
          description: رد من نوغ
          content:
            application/json:
              schema:
                type: object
                properties:
                  response:
                    type: string
```

4. اضغط على "Test" للتأكد من أن Schema صحيح

### 4. إعدادات الأمان / Security Settings

في قسم "Authentication":
- اختر "None" للوقت الحالي (إذا كنت تريد إضافة مصادقة لاحقاً، يمكنك استخدام API Key)

### 5. إعدادات الخصوصية / Privacy Settings

في قسم "Privacy":
- اختر "Only me" للاختبار
- بعد التأكد من العمل، يمكنك تغييرها إلى "Anyone with a link" أو "Public"

### 6. احفظ / Save

اضغط على "Create" في أعلى اليمين

---

## كيفية الاستخدام / How to Use

### أمثلة على الأسئلة / Example Questions:

**بالعربية:**
```
ما هي حالة النظام الحكومي؟
من هم الوزراء المتاحون في النظام؟
أريد التحدث مع وزير المالية
كيف يعمل نظام نوغ؟
ما هي آخر التحديثات؟
```

**In English:**
```
What is the status of the government system?
List all available ministers
I want to talk to the Finance Minister
How does the Noogh system work?
What are the latest updates?
```

---

## التحقق من الاتصال / Testing the Connection

عندما تسأل GPT أي سؤال، سيقوم تلقائياً بـ:
1. الاتصال بـ https://nooogh.com
2. استدعاء الـ API endpoints المناسبة
3. عرض النتائج لك

يمكنك رؤية استدعاءات الـ API في أسفل رد ChatGPT (سيظهر "Used Noogh Unified AI System API")

---

## ملاحظات مهمة / Important Notes

### الفرق بين MCP و Actions / Difference between MCP and Actions:

| Feature | MCP (Claude) | Actions (ChatGPT) |
|---------|-------------|-------------------|
| **Purpose** | Protocol for local tools | Web API integration |
| **Connection** | Stdio/SSE | HTTP REST API |
| **Setup** | Claude Desktop config | Custom GPT configuration |
| **Authentication** | Local | API Key (optional) |
| **Best For** | File operations, local tools | Web services, APIs |

### إذا كنت تريد استخدام MCP / If you want to use MCP:

MCP يعمل فقط مع **Claude Desktop**:
1. قم بتحميل Claude Desktop من https://claude.ai/download
2. أضف تكوين MCP إلى ملف `claude_desktop_config.json`
3. شغل `mcp_server.py` في وضع stdio

---

## استكشاف الأخطاء / Troubleshooting

### المشكلة: ChatGPT لا يستطيع الاتصال بالـ API
**الحل:**
1. تأكد أن الخادم يعمل على https://nooogh.com
2. تحقق من أن Cloudflare Tunnel يعمل
3. تأكد أن CORS مفعل في API

### المشكلة: Actions لا تظهر
**الحل:**
1. تأكد من صحة YAML schema
2. تحقق من أن الـ server URL صحيح: `https://nooogh.com`
3. اضغط "Test" في صفحة Actions

### المشكلة: أخطاء Authentication
**الحل:**
1. إذا كنت تستخدم API Key، تأكد من إضافته في Headers
2. وإلا، اتركه على "None"

---

## الخطوات التالية / Next Steps

بعد إعداد ChatGPT Actions بنجاح، يمكنك:

1. **إضافة المزيد من Endpoints:**
   - `/brain/status` - حالة الدماغ
   - `/trading/signals` - إشارات التداول
   - `/reports/daily` - التقارير اليومية

2. **تحسين الأمان:**
   - إضافة API Key authentication
   - تفعيل rate limiting
   - إضافة request validation

3. **توسيع الوظائف:**
   - إضافة WebSocket support للبيانات الحية
   - ربط قاعدة البيانات
   - إضافة ML models endpoints

---

## روابط مفيدة / Useful Links

- **Dashboard:** https://nooogh.com
- **API Docs:** https://nooogh.com/docs
- **Health Check:** https://nooogh.com/health
- **OpenAPI Schema:** https://nooogh.com/openapi.json

</div>

---

## Summary (English)

This guide explains how to connect the Noogh Unified AI System to ChatGPT using **Actions** instead of MCP (which only works with Claude Desktop).

**Key Points:**
- MCP = Claude Desktop only
- ChatGPT Actions = Web API integration
- Schema file: `chatgpt_schema.yaml`
- Server: https://nooogh.com

**Quick Setup:**
1. Create Custom GPT
2. Add the OpenAPI schema from above
3. Test with questions about ministers and system status
4. All API calls will be made to https://nooogh.com

**The system is ready to use!**
