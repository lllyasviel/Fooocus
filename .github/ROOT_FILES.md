# المجلد الجذر - Root Directory

## فهرس الملفات

| الملف | الوظيفة |
|-------|---------|
| `fooocus_version.py` | رقم الإصدار |
| `shared.py` | المتغيرات المشتركة |
| `args_manager.py` | إدارة معاملات CLI |
| `launch.py` | نقطة الدخول الرئيسية |
| `build_launcher.py` | إنشاء ملفات التشغيل |
| `entry_with_update.py` | نقطة الدخول مع التحديث |
| `webui.py` | بناء واجهة Gradio |
| `experiments_expansion.py` | تجربة التوسيع |
| `experiments_face.py` | تجربة قص الوجوه |
| `experiments_interrogate.py` | تجربة الوصف |
| `experiments_mask_generation.py` | تجربة توليد القناع |

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

**الهيكل الرئيسي:**
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

## مخطط العلاقات للمجلد الجذر

```
launch.py ← entry_with_update.py
    ├── args_manager.py
    ├── build_launcher.py
    └── webui.py
        ├── shared.py
        └── fooocus_version.py
```

---

**آخر تحديث:** 2026
**الإصدار:** 2.5.5
