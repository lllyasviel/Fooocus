---
### 📄 مسار الملف الافتراضي: docs/tests/test_utils.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
اختبارات LoRA parsing لدالة modules.util.parse_lora_references_from_prompt (137 سطر). تتحقق من: تحليل رموز `<lora:name:weight>`، احترام الحد الأقصى (5)، أولوية LoRAs المدخلة يدوياً، التعامل مع LoRAs المتجاورة بدون مسافات، إزالة التكرار، và التعامل مع الأخطاء.

الاختبارات الرئيسية:
- تحليل LoRA واحد ومتعدد
- الحد الأقصى 5 LoRAs
- أولوية LoRAs UI على prompt LoRAs
- إزالة performance LoRA (Lightning/Hyper-SD/Extreme Speed)
- معالجة syntax غير صالح

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **TestParseLoraReferences**: مجموعة اختبارات unittest.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: modules/util.py, unittest
- **يُستخدم بواسطة**: لا يُستخدم مباشرة - اختبارات فقط
- **الأثر البرمجي**: يضمن صحة تحليل LoRA
