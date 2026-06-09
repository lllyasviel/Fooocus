---
### 📄 docs/extras/censor.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
فحص أمان NSFW للصور المُولّدة. يُحمّل StableDiffusionSafetyChecker من transformers عند الحاجة (lazy loading)، ويُفحص الصور بحثاً عن محتوى غير لائق. يُدمَج مع model_management لتحميل النموذج على الجهاز المناسب (CUDA/CPU).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class Censor**: يدير دورة حياة نموذج فحص الأمان.
  - `__init__(self)`: تهيئة مراجع النموذج وإعدادات الجهاز.
  - `init(self)`: تحميل كسول للنموذج من config (safety_checker/FP16)، CLIPImageProcessor، و model_patcher.
  - `censor(self, x)`: تمرير الصور عبر safety_checker، إزالة الصور غير اللائقة، إعادة القائمة المصفّاة.
- **default_censor**: كائن افتراضي من Censor().censor جاهز للاستخدام المباشر.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: ldm_patched/modules/model_management.py, modules/config.py, extras/safety_checker/models/safety_checker.py, ldm_patched/modules/model_patcher.py, transformers (CLIPConfig, CLIPImageProcessor)
- **يُستدعى بواسطة**: modules/async_worker.py (للفلترة قبل عرض النتيجة)
- **الأثر البرمجي**: تعطيل هذا الملف يسمح بعرض محتوى NSFW دون فلترة
