# Fooocus - دليل مسح الكود المصدري الشامل
## Comprehensive Repository Walkthrough

**للمزيد من التفاصيل، راجع:**
- [map-road.md](map-road.md) - خريطة طريق Fooocus
- [ROOT_FILES.md](ROOT_FILES.md) - الملفات الجذرية

---

# المجلد الجذر (Root Directory)

---

## 📄 fooocus_version.py

### 1. الوظيفة الأساسية (Core Function)
يحتوي على متغير واحد فقط يحدد رقم إصدار Fooocus الحالي.

### 2. التفاصيل الفنية والبرمجية
```python
version = '2.5.5'
```
- لا يوجد imports
- لا يوجد دوال أو classes
- فقط متغير نصي واحد

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `launch.py`, `webui.py`, `meta_parser.py`
- **لا يستدعي أي ملف**
- **التأثير عند التعديل:** تغيير الرقم فقط - لا يؤثر على الوظائف

---

## 📄 shared.py

### 1. الوظيفة الأساسية (Core Function)
يحتوي على متغير مشترك واحد لتخزين كائن الجذر لـ Gradio.

### 2. التفاصيل الفنية والبرمجية
```python
gradio_root = None
```
- **`gradio_root`**: كائن `gr.Blocks` الرئيسي للواجهة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py` (لتخزين الكائن), `async_worker.py` (للوصول للـ URL)
- **لا يستدعي أي ملف**
- **التأثير عند التعديل:** إذا حُذف، سيفشل الوصول للـ Gradio app

---

## 📄 args_manager.py

### 1. الوظيفة الأساسية (Core Function)
يُوسع معاملات CLI من `ldm_patched/modules/args_parser.py` بخيارات إضافية خاصة بـ Fooocus.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `ldm_patched.modules.args_parser` - المعاملات الأساسية

**الخيارات المضافة:**
- `--share` - مشاركة على Gradio
- `--preset` - تطبيق إعداد مسبق
- `--language` - ترجمة الواجهة
- `--disable-offload-from-vram` - تعطيل التفريغ من VRAM
- `--theme` - سمة الواجهة
- `--disable-image-log` - تعطيل تسجيل الصور
- `--disable-metadata` - تعطيل البيانات الوصفية
- `--always-download-new-model` - تحميل النماذج دائماً

**القيم الافتراضية:**
```python
args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)
```

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `launch.py`, `webui.py`, وجميع ملفات modules/
- **يستدعي:** `ldm_patched/modules/args_parser.py`
- **التأثير عند التعديل:** إضافة/حذف خيارات CLI

---

## 📄 launch.py

### 1. الوظيفة الأساسية (Core Function)
نقطة الدخول الرئيسية للمشروع - يدير البيئة وتثبيت ال依赖يات وتحميل النماذج وتشغيل WebUI.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `ssl`, `os`, `sys`, `platform`
- `fooocus_version`, `build_launcher`
- `modules.launch_util` - دوال التثبيت
- `modules.model_loader` - تحميل الملفات
- `modules.config` - الإعدادات

**الدوال الرئيسية:**
- `prepare_environment()`: تثبيت PyTorch, xformers, requirements
- `download_models()`: تحميل النماذج من HuggingFace
- `ini_args()`: تحليل معاملات CLI

**تدفق التنفيذ:**
```
1. تعيين المتغيرات البيئية
2. prepare_environment()
3. build_launcher()
4. ini_args()
5. تحميل النماذج
6. from webui import *
```

### 3. شبكة العلاقات ودورة الحياة
- **يبدأ التنفيذ أولاً** في المشروع
- **يستدعي:** `build_launcher.py`, `modules/config.py`, `webui.py`
- **لا يستدعيه ملف آخر** (هو نقطة البداية)
- **التأثير عند التعديل:** قد يكسر بدء التشغيل

---

## 📄 build_launcher.py

### 1. الوظيفة الأساسية (Core Function)
يُنشئ ملفات `.bat` للتشغيل السريع في Windows.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `os`

**الدوال:**
- `build_launcher()`: تُنشئ `run.bat`, `run_anime.bat`, `run_realistic.bat`

**المنطق:**
- يتحقق من وجود `python_embeded/`
- يُنشئ ملفات batch لكل preset

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `launch.py`
- **لا يستدعي أي ملفات أخرى**
- **التأثير عند التعديل:** لا يؤثر على التشغيل الأساسي

---

## 📄 entry_with_update.py

### 1. الوظيفة الأساسية (Core Function)
نقطة دخول بديلة تُحدّث المشروع تلقائياً من Git قبل التشغيل.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `pygit2` - للتحكم في Git

**المنطق:**
1. يتصل بـ remote repository
2. يتحقق من وجود تحديثات
3. يقوم بـ fast-forward merge إذا أمكن
4. يستدعي `launch.py`

### 3. شبكة العلاقات ودورة الحياة
- **يُستخدم عند التحديث التلقائي**
- **يستدعي:** `launch.py`
- **التأثير عند التعديل:** قد يكسر آلية التحديث

---

## 📄 webui.py

### 1. الوظيفة الأساسية (Core Function)
يبني واجهة Gradio الكاملة ويربط الأحداث بالدوال الخلفية.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `gradio` - بناء الواجهة
- `modules.async_worker` - معالج التوليد
- `modules.config` - الإعدادات
- `modules.flags` - الأعلام والثوابت
- `modules.html` - HTML المخصص
- `modules.sdxl_styles` - الأنماط

**الهيك الرئيسي:**
```python
with gr.Blocks():
    # العمود الرئيسي (2:1)
    # - progress_window: معاينة
    # - gallery: النتائج
    # - prompt: حقل البرومبت
    
    # العمود المتقدم
    # - Settings tab
    # - Styles tab
    # - Models tab
    # - Advanced tab
```

**الدوال الرئيسية:**
- `get_task()`: تحويل المدخلات إلى AsyncTask
- `generate_clicked()`: دورة حياة التوليد
- `inpaint_mode_change()`: تغيير وضع Inpaint

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `launch.py` (آخر سطر)
- **يستدعي:** `modules/async_worker.py`, `modules/config.py`, `modules/flags.py`
- **التأثير عند التعديل:** تغيير واجهة المستخدم بالكامل

---

## 📄 experiments_expansion.py

### 1. الوظيفة الأساسية (Core Function)
تجربة محرك توسيع البرومبت (Fooocus Expansion).

### 2. التفاصيل الفنية والبرمجية
```python
from modules.expansion import FooocusExpansion
expansion = FooocusExpansion()
text = 'a handsome man'
for i in range(64):
    print(expansion(text, seed=i))
```

### 3. شبكة العلاقات ودورة الحياة
- **ملف تجريبي** - لا يُستخدم في الإنتاج
- **يستدعي:** `modules/expansion.py`

---

## 📄 experiments_face.py

### 1. الوظيفة الأساسية (Core Function)
تجربة قص الوجوه.

### 2. التفاصيل الفنية والبرمجية
```python
import extras.face_crop as cropper
img = cv2.imread('lena.png')
result = cropper.crop_image(img)
```

### 3. شبكة العلاقات ودورة الحياة
- **ملف تجريبي** - لا يُستخدم في الإنتاج
- **يستدعي:** `extras/face_crop.py`

---

## 📄 experiments_interrogate.py

### 1. الوظيفة الأساسية (Core Function)
تجربة التحقيق الوصفي للصور (BLIP + WD14).

### 2. التفاصيل الفنية والبرمجية
```python
from extras.interrogate import default_interrogator
from extras.wd14tagger import default_interrogator
```

### 3. شبكة العلاقات ودورة الحياة
- **ملف تجريبي** - لا يُستخدم في الإنتاج
- **يستدعي:** `extras/interrogate.py`, `extras/wd14tagger.py`

---

## 📄 experiments_mask_generation.py

### 1. الوظيفة الأساسية (Core Function)
تجربة توليد القناع باستخدام SAM + GroundingDINO.

### 2. التفاصيل الفنية والبرمجية
```python
from extras.inpaint_mask import SAMOptions, generate_mask_from_image
sam_options = SAMOptions(dino_prompt='eye', ...)
mask_image, _, _, _ = generate_mask_from_image(image, sam_options=sam_options)
```

### 3. شبكة العلاقات ودورة الحياة
- **ملف تجريبي** - لا يُستخدم في الإنتاج
- **يستدعي:** `extras/inpaint_mask.py`

---

# مجلد modules/

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
- **التأثير عند تعديل:** تغيير العمليات الحسابية

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
- **التأثير عند تعديل:** تعطيل دعم LoRA

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
- **التأثير عند تعديل:** تعطيل حفظ/قراءة البيانات الوصفية

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
- **التأثير عند تعديل:** تعطيل تسجيل الصور

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
- **التأثير عند تعديل:** تعطيل الأنماط

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
- **التأثير عند تعديل:** تعطيل فرز الأنماط

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
- **التأثير عند تعديل:** تعطيل تكبير الصور

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
- `patched_SDClipModel__init__()`初始化 CLIP
- `patched_SDClipModel_forward()`:Forward pass
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

# مجلد extras/

---

## 📄 extras/censor.py

### 1. الوظيفة الأساسية (Core Function)
فحص الصور للمحتوى NSFW وحجبها.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `Censor`: فاحص NSFW

**الدوال:**
- `censor()`: فحص وحجب الصور

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل فحص NSFW

---

## 📄 extras/expansion.py

### 1. الوظيفة الأساسية (Core Function)
محرك توسيع البرومبت باستخدام GPT2.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `FooocusExpansion`: محرك التوسيع

**الدوال:**
- `__call__()`: توسيع البرومبت

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`, `default_pipeline.py`
- **التأثير عند التعديل:** تعطيل توسيع البرومبت

---

## 📄 extras/face_crop.py

### 1. الوظيفة الأساسية (Core Function)
قص الوجوه من الصور.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `crop_image()`: قص الوجه الأول

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل قص الوجوه

---

## 📄 extras/interrogate.py

### 1. الوظيفة الأساسية (Core Function)
وصف الصور باستخدام نموذج BLIP.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `Interrogator`: واصف الصور

**الدوال:**
- `interrogate()`: وصف صورة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py` (via describe tab)
- **التأثير عند التعديل:** تعطيل الوصف التلقائي

---

## 📄 extras/wd14tagger.py

### 1. الوظيفة الأساسية (Core Function)
وصف الصور الأنمي باستخدام WD14 Tagger.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `default_interrogator()`: وصف صورة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py` (via describe tab)
- **التأثير عند التعديل:** تعطيل وصف الأنمي

---

## 📄 extras/resampler.py

### 1. الوظيفة الأساسية (Core Function)
نموذج Resampler لـ IP-Adapter.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `PerceiverAttention`: انتباه Perceiver
- `Resampler`: نموذج Resampler

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `ip_adapter.py`
- **التأثير عند تعديل:** تعطيل IP-Adapter

---

## 📄 extras/preprocessors.py

### 1. الوظيفة الأساسية (Core Function)
المعالجة المسبقة للصور (Canny, CPDS).

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `canny_pyramid()`: Canny مع هرمي
- `cpds()`: CPDS preprocessing

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل المعالجة المسبقة

---

## 📄 extras/ip_adapter.py

### 1. الوظيفة الأساسية (Core Function)
تطبيق IP-Adapter على النموذج.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `ImageProjModel`, `To_KV`, `IPAdapterModel`

**الدوال:**
- `load_ip_adapter()`: تحميل IP-Adapter
- `preprocess()`: معالجة الصورة
- `patch_model()`: تطبيق على النموذج

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل IP-Adapter

---

## 📄 extras/inpaint_mask.py

### 1. الوظيفة الأساسية (Core Function)
توليد قناع Inpaint باستخدام SAM + GroundingDINO.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `SAMOptions`: خيارات SAM

**الدوال:**
- `generate_mask_from_image()`: توليد القناع
- `optimize_masks()`: تحسين القناع

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`, `async_worker.py`
- **التأثير عند التعديل:** تعطيل توليد القناع

---

## 📄 extras/vae_interpose.py

### 1. الوظيفة الأساسية (Core Function)
تحويل Latent من SDXL إلى SD1.5.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `InterposerModel`: نموذج التحويل

**الدوال:**
- `parse()`: تحويل Latent

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `default_pipeline.py`
- **التأثير عند التعديل:** تعطيل التحويل

---

# مجلد ldm_patched/

---

## 📄 ldm_patched/modules/sd.py

### 1. الوظيفة الأساسية (Core Function)
تحميل النماذج والتعامل مع 파일ات الأوزان.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `CLIP`: نموذج CLIP

**الدوال:**
- `load_model_weights()`: تحميل الأوزان
- `load_clip_weights()`: تحميل أوزان CLIP
- `load_lora_for_models()`: تحميل LoRA
- `load_checkpoint_guess_config()`: تحميل checkpoint

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `modules/core.py`
- **التأثير عند التعديل:** تعطيل تحميل النماذج

---

## 📄 ldm_patched/modules/samplers.py

### 1. الوظيفة الأساسية (Core Function)
تنفيذ خوارزميات العينات (Sampling).

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `KSampler`: العينات الرئيسية

**الدوال:**
- `sample()`: تنفيذ العينات
- `calc_cond_uncond_batch()`: حساب conditioning

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `modules/sample.py`, `modules/sample_hijack.py`
- **التأثير عند التعديل:** تعطيل العينات

---

## 📄 ldm_patched/modules/sample.py

### 1. الوظيفة الأساسية (Core Function)
إعداد وتنفيذ عملية التوليد.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `prepare_noise()`: تحضير الضوضاء
- `prepare_mask()`: تحضير القناع
- `sample()`: التنفيذ الرئيسي

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `modules/core.py`
- **التأثير عند التعديل:** تعطيل التوليد

---

## 📄 ldm_patched/modules/model_management.py

### 1. الوظيفة الأساسية (Core Function)
إدارة الذاكرة وتحميل النماذج على GPU.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `VRAMState`: حالات VRAM
- `CPUState`: حالات المعالج
- `LoadedModel`: نموذج محمل

**الدوال:**
- `get_torch_device()`: الحصول على الجهاز
- `load_models_gpu()`: تحميل النماذج
- `free_memory()`: تحرير الذاكرة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** جميع ملفات ldm_patched
- **التأثير عند التعديل:** تعطيل إدارة الذاكرة

---

## 📄 ldm_patched/modules/model_patcher.py

### 1. الوظيفة الأساسية (Core Function)
تطبيق الـ patches على النماذج.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `ModelPatcher`: مُطبّق الـ patches

**الدوال:**
- `clone()`: نسخ النموذج
- `add_patches()`: إضافة patches
- `set_model_patch()`: تعيين patch

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** جميع ملفات ldm_patched
- **التأثير عند التعديل:** تعطيل نظام الـ patches

---

## 📄 ldm_patched/modules/args_parser.py

### 1. الوظيفة الأساسية (Core Function)
تحليل معاملات CLI الأساسية.

### 2. التفاصيل الفنية والبرمجية
**المعاملات:**
- `--listen`, `--port`: خادم الشبكة
- `--always-cpu`, `--always-low-vram`: إدارة الذاكرة
- `--attention-split`, `--attention-quad`: الانتباه
- `--all-in-fp32`, `--all-in-fp16`: الدقة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `args_manager.py`
- **التأثير عند التعديل:** تغيير معاملات CLI

---

## 📄 ldm_patched/modules/options.py

### 1. الوظيفة الأساسية (Core Function)
تفعيل/تعطيل تحليل المعاملات.

### 2. التفاصيل الفنية والبرمجية
```python
args_parsing = False
def enable_args_parsing(enable=True):
    global args_parsing
    args_parsing = enable
```

### 3. شبكة العلاقات ودورة الحياة
- **التأثير عند التعديل:** تغيير سلوك تحليل المعاملات

---

# مجلد tests/

---

## 📄 tests/__init__.py

### 1. الوظيفة الأساسية (Core Function)
إضافة مسار modules إلى sys.path للاختبارات.

### 2. التفاصيل الفنية والبرمجية
```python
sys.path.append(pathlib.Path(f'{__file__}/../modules').parent.resolve())
```

### 3. شبكة العلاقات ودورة الحياة
- **ضروري لتشغيل الاختبارات**

---

## 📄 tests/test_utils.py

### 1. الوظيفة الأساسية (Core Function)
اختبارات لدوال `modules/util.py`.

### 2. التفاصيل الفنية والبرمجية
**الاختبارات:**
- `test_can_parse_tokens_with_lora()`: اختبار تحليل LoRA
- `test_can_parse_tokens_and_strip_performance_lora()`: اختبار إزالة LoRA

### 3. شبكة العلاقات ودورة الحياة
- **يختبر:** `modules/util.py`

---

## 📄 tests/test_extra_utils.py

### 1. الوظيفة الأساسية (Core Function)
اختبارات لدوال `modules/extra_utils.py`.

### 2. التفاصيل الفنية والبرمجية
**الاختبارات:**
- `test_try_eval_env_var()`: اختبار تقييم متغيرات البيئة

### 3. شبكة العلاقات ودورة الحياة
- **يختبر:** `modules/extra_utils.py`

---

# ملخصعلاقات الاعتماد (Dependency Map)

```
launch.py
    ├── args_manager.py
    ├── modules/config.py
    └── webui.py
        ├── modules/async_worker.py
        │   ├── modules/default_pipeline.py
        │   │   └── modules/core.py
        │   └── modules/patch.py
        └── modules/sdxl_styles.py
```

---

**آخر تحديث:** 2026
**الإصدار:** 2.5.5
