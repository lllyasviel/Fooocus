---
### 📄 مسار الملف الافتراضي: docs/root/README.md

# Fooocus Technical Documentation

## نظرة عامة
Fooocus هو أداة لتوليد الصور باستخدام Stable Diffusion XL (SDXL). تم تصميمه ليكون سهل الاستخدام مع الحفاظ على الميزات المتقدمة.

## هيكل التوثيق

### المجلدات الموثقة:
1. **[docs/root/](root/)** - الملفات الرئيسية (launch.py, webui.py, shared.py, args_manager.py, etc.)
2. **[docs/modules/](modules/)** - وحدات Fooocus الأساسية (config, core, async_worker, patch, flags, etc.)
3. **[docs/ldm_patched/](ldm_patched/)** - محرك ldm_patched (modules, contrib, ldm, k_diffusion, pfn, etc.)
4. **[docs/extras/](extras/)** - الإضافات (censor, expansion, ip_adapter, preprocessors, BLIP, facexlib, etc.)
5. **[docs/css/](css/)** - تنسيقات CSS
6. **[docs/javascript/](javascript/)** - ملفات JavaScript
7. **[docs/language/](language/)** - ملفات التوطين
8. **[docs/presets/](presets/)** - القوالب المسبقة
9. **[docs/sdxl_styles/](sdxl_styles/)** - أنماط SDXL
10. **[docs/wildcards/](wildcards/)** - ملفات Wildcard
11. **[docs/models/](models/)** - هيكل مجلد النماذج
12. **[docs/tests/](tests/)** - اختبارات الوحدات

## نقاط الدخول الرئيسية
- **نقطة الدخول الفعلية**: `entry_with_update.py` (مع تحديث Git)
- **نقطة التشغيل**: `launch.py` (إعداد البيئة + تحميل النماذج)
- **واجهة المستخدم**: `webui.py` (بناء Gradio UI)
- **عامل التوليد**: `modules/async_worker.py` (خيط الخلفية)
- **الترقيعات**: `modules/patch.py` (تحسينات Fooocus على ldm_patched)

## مسار البيانات
```
User Input (webui.py) → AsyncTask (async_worker.py) → Pipeline (default_pipeline.py) → Core (core.py) → ldm_patched → Output
```
