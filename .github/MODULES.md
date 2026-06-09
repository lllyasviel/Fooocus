# مجلد modules/ - الوحدات الأساسية

## فهرس الملفات

| الملف | الوظيفة |
|-------|---------|
| `__init__.py` | تعريف المجلد كـ package |
| `anisotropic.py` | فلاتر Blur شعاعية |
| `auth.py` | نظام المصادقة |
| `constants.py` | الثوابت |
| `html.py` | إنشاء HTML |
| `extra_utils.py` | دوال مساعدة |
| `hash_cache.py` | تخزين مؤقت للـ hashes |
| `gradio_hijack.py` | تعديل مكونات Gradio |
| `localization.py` | الترجمة |
| `model_loader.py` | تحميل الملفات |
| `ui_gradio_extensions.py` | حقن JavaScript/CSS |
| `ops.py` | تبديل العمليات الحسابية |
| `lora.py` | تحميل LoRA |
| `meta_parser.py` | تحليل البيانات الوصفية |
| `private_logger.py` | تسجيل الصور |
| `sdxl_styles.py` | أنماط SDXL |
| `style_sorter.py` | فرز الأنماط |
| `upscaler.py` | تكبير الصور |
| `sample_hijack.py` | تعديل العينات |
| `patch_precision.py` | تعديل الدقة |
| `patch_clip.py` | تعديل CLIP |
| `inpaint_worker.py` | معالج Inpaint |
| `default_pipeline.py` | خط الأنابيب الافتراضي |
| `core.py` | النواة الحسابية |
| `config.py` | إدارة الإعدادات |
| `async_worker.py` | المعالج غير المتزامن |
| `util.py` | دوال مساعدة متنوعة |

---

## 📄 modules/__init__.py

### 1. الوظيفة الأساسية (Core Function)
ملف فارغ - يجعل المجلد package Python.

### 2. التفاصيل الفنية والبرمجية
لا يوجد محتوى.

### 3. شبكة العلاقات ودورة الحياة
- **ضروري لاستيراد الملفات من هذا المجلد**

---

## 📄 modules/anisotropic.py

### 1. الوظيفة الأساسية (Core Function)
تطبيق فلاتر Blur شعاعية (Anisotropic/Bilateral) على الصور.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `torch`

**الدوال الرئيسية:**
- `bilateral_blur()`: ضبابية ثنائية الجانب
- `joint_bilateral_blur()`: ضبابية ثنائية مع إرشاد
- `adaptive_anisotropic_filter()`: فلتر تكيّفي

**الفئات:**
- `BilateralBlur`: وحدة PyTorch للضبابية
- `JointBilateralBlur`: وحدة مع إرشاد

### 3. شبكة العلاقات ودورة الحياة
- **لا يستدعيه ملف آخر مباشرة** (يُستخدم محلياً)
- **التأثير عند التعديل:** تغيير خوارزميات الضبابية

---

## 📄 modules/auth.py

### 1. الوظيفة الأساسية (Core Function)
نظام المصادقة البسيط لحماية الوصول.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `json`, `hashlib`, `modules.constants`

**الدوال:**
- `load_auth_data()`: تحميل بيانات المصادقة من JSON
- `check_auth()`: التحقق من اسم المستخدم وكلمة المرور
- `auth_list_to_dict()`: تحويل القاموس

**المنطق:**
- يقرأ من `auth.json`
- يستخدم SHA256 لكلمة المرور

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`
- **يستدعي:** `modules/constants.py`
- **التأثير عند التعديل:** تعطيل/تفعيل المصادقة

---

## 📄 modules/constants.py

### 1. الوظيفة الأساسية (Core Function)
تعريف الثوابت الأساسية للمشروع.

### 2. التفاصيل الفنية والبرمجية
```python
MIN_SEED = 0
MAX_SEED = 2**63 - 1
AUTH_FILENAME = 'auth.json'
```

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `auth.py`, `async_worker.py`, `patch.py`
- **لا يستدعي أي ملف**
- **التأثير عند التعديل:** تغيير النطاقات

---

## 📄 modules/html.py

### 1. الوظيفة الأساسية (Core Function)
إنشاء عناصر HTML للتقدم في الواجهة.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `make_progress_html(number, text)`: إنشاء شريط التقدم

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`, `async_worker.py`
- **التأثير عند التعديل:** تغيير مظهر شريط التقدم

---

## 📄 modules/extra_utils.py

### 1. الوظيفة الأساسية (Core Function)
دوال مساعدة عامة للتعامل مع الملفات والمجلدات.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `makedirs_with_log()`: إنشاء مجلد مع تسجيل
- `get_files_from_folder()`: الحصول على ملفات من مجلد
- `try_eval_env_var()`: تقييم متغيرات البيئة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `config.py`, `sdxl_styles.py`
- **التأثير عند التعديل:** تعطيل دوال المساعدة

---

## 📄 modules/hash_cache.py

### 1. الوظيفة الأساسية (Core Function)
نظام تخزين مؤقت للـ hashes لتسريع التحميل.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `json`, `concurrent.futures`, `modules.util`

**الدوال:**
- `sha256_from_cache()`: حساب/تخزين hash
- `load_cache_from_file()`: تحميل من ملف
- `save_cache_to_file()`: حفظ في ملف
- `rebuild_cache()`: إعادة بناء الكاش

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `launch.py`, `meta_parser.py`
- **يستدعي:** `modules/util.py`
- **التأثير عند التعديل:** بطء حساب الـ hashes

---

## 📄 modules/gradio_hijack.py

### 1. الوظيفة الأساسية (Core Function)
 تعديل (hijack) مكون Image في Gradio لدعم ميزات إضافية.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `gradio`, `PIL`, `numpy`

**الفئات:**
- `Image`: نسخة معدّلة من `gradio.Image` مع دعم sketch و masks

**التعديلات:**
- تجاوز timeout في `asyncio.wait_for`
- تتبع جميع المكونات في `all_components`

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`
- **التأثير عند التعديل:** تغيير سلوك مكونات الصور

---

## 📄 modules/localization.py

### 1. الوظيفة الأساسية (Core Function)
نظام الترجمة للواجهة المتعددة اللغات.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `localization_js()`: تحميل ملف الترجمة وإرجاع JavaScript
- `dump_english_config()`: تصدير النصوص الإنجليزية

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `ui_gradio_extensions.py`
- **التأثير عند تعديل:** تعطيل الترجمة

---

## 📄 modules/model_loader.py

### 1. الوظيفة الأساسية (Core Function)
تحميل الملفات من URLs مع دعم الكاش المحلي.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `load_file_from_url()`: تحميل ملف من URL مع التخزين المؤقت

**المنطق:**
- يتحقق من الملف المحلي أولاً
- يستخدم `torch.hub.download_url_to_file`
- يدعم HuggingFace Mirror

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `launch.py`, `interrogate.py`, `wd14tagger.py`
- **التأثير عند التعديل:** تعطيل تحميل النماذج

---

## 📄 modules/ui_gradio_extensions.py

### 1. الوظيفة الأساسية (Core Function)
حقن ملفات JavaScript و CSS في واجهة Gradio.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `javascript_html()`: إنشاء tags script
- `css_html()`: إنشاء tags link
- `reload_javascript()`: حقن الملفات في HTML

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`
- **يستدعي:** `modules/localization.py`
- **التأثير عند التعديل:** تعطيل JavaScript/CSS المخصص

---

## 📄 modules/ops.py

### 1. الوظيفة الأساسية (Core Function)
تبديل العمليات الحسابية مؤقتاً (context manager).

### 2. التفاصيل الفنية والبرمجية
```python
@contextlib.contextmanager
def use_patched_ops(operations):
    # يستبدل Linear, Conv2d, GroupNorm, LayerNorm مؤقتاً
```

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `patch_clip.py`, `ip_adapter.py`
- **التأثير عند التعديل:** تغيير العمليات الحسابية

---

## 📄 modules/lora.py

### 1. الوظيفة الأساسية (Core Function)
مطابقة وتحميل ملفات LoRA بأنواع مختلفة.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `match_lora()`: مطابقة أسماء المفاتيح مع دعم أنواع متعددة

**أنواع LoRA المدعومة:**
- LoRA العادي
- LoHa
- LoKr
- GLoRA
- Diff

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `core.py`
- **التأثير عند التعديل:** تعطيل دعم LoRA

---

## 📄 modules/meta_parser.py

### 1. الوظيفة الأساسية (Core Function)
تحليل وحفظ البيانات الوصفية للصور بتنسيقات مختلفة.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `MetadataParser`: الفئة الأساسية المجردة
- `A1111MetadataParser`: تنسيق A1111/Civitai
- `FooocusMetadataParser`: تنسيق Fooocus الخاص

**الدوال:**
- `load_parameter_button_click()`: تحميل المعاملات من صورة
- `read_info_from_image()`: قراءة البيانات من صورة
- `get_exif()`: إنشاء بيانات EXIF

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`, `private_logger.py`, `async_worker.py`
- **يستدعي:** `modules/config.py`, `modules/hash_cache.py`
- **التأثير عند التعديل:** تعطيل حفظ/قراءة البيانات الوصفية

---

## 📄 modules/private_logger.py

### 1. الوظيفة الأساسية (Core Function)
تسجيل الصور المولّدة في ملف HTML خاص.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `log()`: حفظ الصورة وتسجيل البيانات في HTML
- `get_current_html_path()`: الحصول على مسار ملف السجل

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **يستدعي:** `modules/meta_parser.py`
- **التأثير عند التعديل:** تعطيل تسجيل الصور

---

## 📄 modules/sdxl_styles.py

### 1. الوظيفة الأساسية (Core Function)
تحميل وإدارة أنماط SDXL للبرومبت.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `apply_style()`: تطبيق نمط على البرومبت
- `get_random_style()`: الحصول على نمط عشوائي
- `apply_arrays()`: معالجة المصفوفات في البرومبت

**البيانات:**
- `styles`: قاموس الأنماط
- `fooocus_expansion`: اسم نمط Fooocus V2
- `legal_style_names`: الأسماء القانونية

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`, `async_worker.py`
- **يستدعي:** `modules/extra_utils.py`
- **التأثير عند التعديل:** تعطيل الأنماط

---

## 📄 modules/style_sorter.py

### 1. الوظيفة الأساسية (Core Function)
إدارة وفرز الأنماط في الواجهة.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `try_load_sorted_styles()`: تحميل الترتيب المحفوظ
- `sort_styles()`: حفظ الترتيب الجديد
- `search_styles()`: البحث في الأنماط

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`
- **التأثير عند التعديل:** تعطيل فرز الأنماط

---

## 📄 modules/upscaler.py

### 1. الوظيفة الأساسية (Core Function)
تكبير الصور باستخدام ESRGAN.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `perform_upscale()`: تكبير الصورة

**الفئات:**
- `RRDBNet`: نموذج ESRGAN

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`, `inpaint_worker.py`
- **التأثير عند التعديل:** تعطيل تكبير الصور

---

## 📄 modules/sample_hijack.py

### 1. الوظيفة الأساسية (Core Function)
تعديل دوال العينات (Sampling) في ldm_patched لدعم Refiner.

### 2. التفاصيل الفنية والبرمجية
**الدوال الرئيسية:**
- `clip_separate()`: فصل CLIP للـ Refiner
- `sample_hacked()`: دالة العينات المعدّلة
- `calculate_sigmas_scheduler_hacked()`: حساب Sigmas

**التعديلات:**
- يُعدّل `ldm_patched.modules.samplers.sample`
- يُعدّل `ldm_patched.modules.samplers.calculate_sigmas_scheduler`

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `patch.py` (عبر `patch_all()`)
- **يعدّل:** `ldm_patched/modules/samplers.py`
- **التأثير عند التعديل:** تعطيل دعم Refiner

---

## 📄 modules/patch_precision.py

### 1. الوظيفة الأساسية (Core Function)
تعديل دوال Time Embedding و Register Schedule لتطابق Kohya.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `patched_timestep_embedding()`: Time embedding معدّل
- `patched_register_schedule()`: Schedule معدّل
- `patch_all_precision()`: تطبيق التعديلات

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `patch.py`
- **التأثير عند التعديل:** تغيير دقة النموذج

---

## 📄 modules/patch_clip.py

### 1. الوظيفة الأساسية (Core Function)
تعديل نماذج CLIP لتطابق Kohya/A1111.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `patched_encode_token_weights()`: تشفير الأوزان
- `patched_SDClipModel__init__()` initialise CLIP
- `patched_SDClipModel_forward()`: Forward pass
- `patch_all_clip()`: تطبيق جميع التعديلات

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `patch.py`
- **التأثير عند التعديل:** تغيير تشفير النص

---

## 📄 modules/patch.py

### 1. الوظيفة الأساسية (Core Function)
تطبيق تعديلات Fooocus على نموذج diffusion.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `PatchSettings`: إعدادات التعديلات

**الدوال:**
- `calculate_weight_patched()`: حساب الأوزان مع LoRA
- `patch_all()`: تطبيق جميع التعديلات

**أنواع الـ Patches:**
- `diff`, `lora`, `fooocus`, `loha`, `lokr`, `glora`

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **يستدعي:** `patch_precision.py`, `patch_clip.py`, `sample_hijack.py`
- **التأثير عند التعديل:** تغيير سلوك النموذج بالكامل

---

## 📄 modules/default_pipeline.py

### 1. الوظيفة الأساسية (Core Function)
إدارة خط أنابيب التوليد الافتراضي.

### 2. التفاصيل الفنية والبرمجية
**المتغيرات الرئيسية:**
- `model_base`, `model_refiner`: النماذج المحملة
- `loaded_ControlNets`: ControlNets المحملة

**الدوال:**
- `refresh_base_model()`: تحديث النموذج الأساسي
- `refresh_refiner_model()`: تحديث Refiner
- `process_diffusion()`: تنفيذ التشتت

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **يستدعي:** `modules/core.py`
- **التأثير عند التعديل:** تعطيل خط الأنابيب

---

## 📄 modules/core.py

### 1. الوظيفة الأساسية (Core Function)
النواة الحسابية - تحميل النماذج وتشغيل التوليد.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `StableDiffusionModel`: نموذج Stable Diffusion
- `VAEApprox`: تقريب VAE للمعاينة

**الدوال:**
- `load_model()`: تحميل نموذج
- `load_controlnet()`: تحميل ControlNet
- `ksampler()`: العينات
- `encode_vae()`, `decode_vae()`: ترميز VAE

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `default_pipeline.py`, `upscaler.py`, `ip_adapter.py`
- **يستدعي:** `ldm_patched/modules/sd.py`
- **التأثير عند التعديل:** تعطيل التوليد بالكامل

---

## 📄 modules/config.py

### 1. الوظيفة الأساسية (Core Function)
إدارة جميع الإعدادات والمسارات في المشروع.

### 2. التفاصيل الفنية والبرمجية
**الدوال الرئيسية:**
- `get_dir_or_set_default()`: الحصول على مسار
- `get_config_item_or_set_default()`: الحصول على إعداد
- `update_files()`: تحديث قوائم الملفات

**المسارات:**
- `paths_checkpoints`, `paths_loras`, `path_embeddings`, `path_vae`, etc.

**الإعدادات الافتراضية:**
- `default_base_model_name`, `default_cfg_scale`, `default_sampler`, etc.

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** جميع ملفات modules تقريباً
- **يستدعي:** `presets/default.json`, `config.txt`
- **التأثير عند التعديل:** تعطيل الإعدادات بالكامل

---

## 📄 modules/async_worker.py

### 1. الوظيفة الأساسية (Core Function)
المعالج غير المتزامن للصور - يدير دورة حياة التوليد.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `AsyncTask`: تمثيل طلب توليد واحد

**الدوال:**
- `worker()`: الحلقة الرئيسية للمعالج
- `process_task()`: معالجة مهمة واحدة
- `apply_vary()`: تطبيق التباين
- `apply_inpaint()`: تطبيق Inpaint

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`
- **يستدعي:** `modules/default_pipeline.py`, `modules/patch.py`, `modules/core.py`
- **التأثير عند التعديل:** تعطيل التوليد بالكامل

---

## 📄 modules/util.py

### 1. الوظيفة الأساسية (Core Function)
دوال مساعدة متنوعة للمشروع.

### 2. التفاصيل الفنية والبرمجية
**الدوال الرئيسية:**
- `sha256()`: حساب hash
- `HWC3()`: تحويل تنسيق الصورة
- `resize_image()`: تغيير حجم الصورة
- `parse_lora_references_from_prompt()`: تحليل LoRA من البرومبت
- `apply_wildcards()`: تطبيق Wildcards

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** معظم ملفات modules
- **التأثير عند التعديل:** تعطيل الدوال المساعدة

---

## مخطط العلاقات لمجلد modules/

```
webui.py
    ├── async_worker.py
    │   ├── default_pipeline.py
    │   │   └── core.py
    │   │       └── lora.py
    │   ├── patch.py
    │   │   ├── patch_precision.py
    │   │   ├── patch_clip.py
    │   │   └── sample_hijack.py
    │   ├── inpaint_worker.py
    │   │   └── upscaler.py
    │   └── private_logger.py
    │       └── meta_parser.py
    ├── config.py
    │   └── extra_utils.py
    ├── sdxl_styles.py
    │   └── extra_utils.py
    ├── html.py
    ├── localization.py
    │   └── ui_gradio_extensions.py
    ├── auth.py
    │   └── constants.py
    └── style_sorter.py
```

---

**آخر تحديث:** 2026
**الإصدار:** 2.5.5
