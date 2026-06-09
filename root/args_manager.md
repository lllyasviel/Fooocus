---
### 📄 مسار الملف الافتراضي: docs/root/args_manager.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
إدارة معاملات سطر الأوامر (CLI Arguments) لأداة Fooocus. يمتد معاملات ldm_patched.modules.args_parser بإضافة معاملات Fooocus المحددة مثل المشاركة (--share)، والقالب (--preset)، واللغة (--language)، وتعطيل التحليلات.

المعاملات المُضافة:
- --share: مشاركة على Gradio
- --preset: تطبيق قالب UI محدد
- --disable-preset-selection: تعطيل اختيار القالب
- --language: ترجمة واجهة المستخدم
- --disable-offload-from-vram: إجبار تحميل النماذج إلى VRAM
- --theme: سمة الواجهة (فاتح/داكن)
- --disable-image-log: منع حفظ الصور
- --disable-analytics: تعطيل التحليلات
- --disable-metadata: تعطيل حفظ البيانات الوصفية
- --disable-preset-download: تعطيل تحميل نماذج القالب
- --disable-enhance-output-sorting: تعطيل ترتيب مخرجات التحسين
- --enable-auto-describe-image: تفعيل وصف الصورة التلقائي
- --always-download-new-model: إجبار تحميل النماذج الجديدة
- --rebuild-hash-cache: إعادة بناء كاش التوقيعات

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- لا يحتوي على دوال أو كلاسات - ملف تعريف معاملات فقط
- **args**: كائن argparse.Namespace يحتوي على جميع المعاملات المُحللة

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: ldm_patched/modules/args_parser.py (الأساسي)
- **يُستخدم بواسطة**: launch.py, webui.py, modules/config.py, modules/async_worker.py, modules/auth.py, modules/patch.py, extras/inpaint_mask.py
- **الأثر البرمجي**: أي معامل جديد هنا يمكن الوصول إليه من أي مكان في المشروع عبر args_manager.args
