# Fooocus - الوثيقة الفنية الشاملة والدليل الهندسي المتكامل

## Fooocus v2.5.5 - دليل هندسي للمطورين

---

## القسم 1: الهندسة المعمارية للمشروع

### 1.1 الفلسفة البرمجية

 Fooocus صُمم وفق فلسفة **"البساطة في الواجهة والتعقيد في الخلفية"**. يهدف المشروع إلى تقديم تجربة مستخدم مبسطة (类 Midjourney) بينما يحتفظ بقوة Stable Diffusion في الخلفية.

**المبادئ الأساسية:**
- **Abstraction Layer**: فصل كامل بين واجهة المستخدم ومحرك التوليد
- **Convention over Configuration**: إعدادات افتراضية مدروسة تقلل الحاجة للتدخل اليدوي
- **Progressive Disclosure**: خيارات متقدمة مخفية بشكل افتراضي但 يمكن فتحها

### 1.2 العلاقة مع ComfyUI (ldm_patched)

مجلد `ldm_patched/` هو في الواقع **fork معدّل** من محرك ComfyUI، يوفر:
- إدارة النماذج والذاكرة (VRAM Management)
- معالجة الـ Samplers والـ Schedulers
- دعم LoRA و ControlNet
- نظام الـ Patches الذي يسمح بتعديل سلوك النموذج دون تغيير الأوزان

**ميزة الدمج:** Fooocus يستفيد من قوة ComfyUI في إدارة الموارد while يضيف طبقة تجريد عالية-level تجعل الاستخدام أسهل.

---

## القسم 2: تشريح هيكل الملفات والمجلدات

### 2.1 المجلدات الرئيسية

```
Fooocus/
├── launch.py              # نقطة الدخول الرئيسية
├── webui.py               # بناء واجهة Gradio
├── args_manager.py        # إدارة معاملات CLI
├── shared.py              # المتغيرات المشتركة
├── build_launcher.py      # إنشاء ملفات التشغيل
├── fooocus_version.py     # رقم الإصدار
├── modules/               # الوحدات الأساسية
├── ldm_patched/           # محرك ComfyUI المعدّل
├── extras/                # الإضافات والمحاضرات
├── presets/               # ملفات الإعدادات المسبقة
├── models/                # النماذج والملفات
├── sdxl_styles/           # أنماط SDXL
├── wildcards/             # ملفات Wildcards
├── outputs/               # مجلد المخرجات
├── css/                   # ملفات التنسيق
├── javascript/            # ملفات JavaScript
├── language/              # ملفات الترجمة
└── tests/                 # الاختبارات
```

### 2.2 ملف launch.py (نقطة الدخول)

**المسؤوليات:**
1. تعيين المتغيرات البيئية (GPU, SSL, Gradio)
2. التحقق من ال依赖يات وتثبيتها
3. تحميل النماذج الافتراضية
4. بدء تشغيل WebUI

**تدفق التنفيذ:**
```
launch.py → prepare_environment() → build_launcher() → ini_args() → download_models() → from webui import *
```

**الكود الرئيسي (lines 70-152):**
- `prepare_environment()`: تثبيت PyTorch, xformers, والمتطلبات
- `build_launcher()`: إنشاء ملفات .bat للتشغيل السريع
- `download_models()`: تحميل النماذج من HuggingFace
- `init_cache()`: إنشاء cache للملفات

### 2.3 ملف webui.py (واجهة المستخدم)

**المسؤوليات:**
- بناء واجهة Gradio الكاملة
- ربط الأحداث بالدوال
- إدارة حالة التوليد

**الهيكل:**
```python
with gr.Blocks():
    with gr.Row():
        # العمود الرئيسي (2:1 ratio)
        with gr.Column(scale=2):
            progress_window    # معاينة مباشرة
            gallery           # النتائج النهائية
            prompt            # حقل البرومبت
        
        # العمود المتقدم
        with gr.Column(scale=1):
            # تبويبات: Settings, Styles, Models, Advanced
```

**الوظائف الرئيسية:**
- `get_task()`: تحويل входات UI إلى AsyncTask
- `generate_clicked()`: دورة حياة التوليد
- `sort_enhance_images()`: ترتيب صور التحسين

### 2.4 مجلد modules/ (الوحدة الأساسية)

#### 2.4.1 async_worker.py (المعالج غير المتزامن)

**المسؤوليات:**
- إنشاء `AsyncTask` لتخزين إعدادات كل طلب
- تنفيذ عملية التوليد في thread منفصل
- إدارة التقدم والنتائج

**بنية AsyncTask:**
```python
class AsyncTask:
    def __init__(self, args):
        self.prompt = args.pop()
        self.negative_prompt = args.pop()
        self.style_selections = args.pop()
        self.performance_selection = Performance(args.pop())
        self.steps = self.performance_selection.steps()
        self.aspect_ratios_selection = args.pop()
        # ... 80+ خاصية
```

**دالة worker():**
- الحلقة الرئيسية: `while not finished:`
- معالجة المعاينة: `flag == 'preview'`
- إرجاع النتائج: `flag == 'results'`
- الإنهاء: `flag == 'finish'`

#### 2.4.2 config.py (إدارة الإعدادات)

**المسؤوليات:**
- تحميل `presets/default.json`
- تحميل `config.txt` (تخصيصات المستخدم)
- تحديد مسارات النماذج

**تسلسل التحميل:**
1. `presets/default.json` (الإعدادات الافتراضية)
2. `config.txt` (تخصيصات المستخدم)
3. متغيرات البيئة

**المسارات الرئيسية:**
```python
paths_checkpoints = ['../models/checkpoints/']
paths_loras = ['../models/loras/']
path_embeddings = '../models/embeddings/'
path_vae = '../models/vae/'
path_upscale_models = '../models/upscale_models/'
path_inpaint = '../models/inpaint/'
```

#### 2.4.3 core.py (النواة الحسابية)

**المسؤوليات:**
- تحميل النماذج: `load_model()`
- معالجة LoRA: `StableDiffusionModel.refresh_loras()`
- ترميز VAE: `encode_vae()`, `decode_vae()`
- العينات: `ksampler()`

**بنية StableDiffusionModel:**
```python
class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision, filename, vae_filename):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.unet_with_lora = unet
        self.clip_with_lora = clip
```

#### 2.4.4 patch.py (نظام الـ Patches)

**المسؤوليات:**
- تعديل سلوك النموذج أثناء التشغيل
- تطبيق Sharpness, ADM Guidance, Adaptive CFG

**PatchSettings:**
```python
class PatchSettings:
    def __init__(self, sharpness, adm_scaler_end, positive_adm_scale, 
                 negative_adm_scale, controlnet_softness, adaptive_cfg):
```

**أنواع الـ Patches:**
- `diff`: فرق الأوزان
- `lora`: تعديلات LoRA
- `fooocus`: تعديلات خاصة بـ Fooocus

#### 2.4.5 default_pipeline.py (خط الأنابيب الافتراضي)

**المسؤوليات:**
- إدارة النماذج المحملة
- تحديث Base Model و Refiner
- تنفيذ `process_diffusion()`

### 2.5 مجلد ldm_patched/ (محرك ComfyUI)

#### 2.5.1 modules/model_management.py

**المسؤوليات:**
- إدارة VRAM والذاكرة
- تحميل/تفريغ النماذج من/إلى GPU

**حالات VRAM:**
```python
class VRAMState(Enum):
    DISABLED = 0    # لا يوجد VRAM
    NO_VRAM = 1     # VRAM منخفض جداً
    LOW_VRAM = 2    # VRAM منخفض
    NORMAL_VRAM = 3 # VRAM عادي
    HIGH_VRAM = 4   # VRAM كبير
    SHARED = 5      #ذاكرة مشتركة
```

**الدوال الرئيسية:**
- `get_torch_device()`: تحديد الجهاز المستخدم
- `load_models_gpu()`: تحميل النماذج إلى GPU
- `free_memory()`: تحرير الذاكرة

#### 2.5.2 modules/args_parser.py

**مسؤول:** تحليل معاملات CLI

**المعاملات الحرجة:**
```python
# إدارة الذاكرة
parser.add_argument("--always-cpu")
parser.add_argument("--always-low-vram")
parser.add_argument("--always-no-vram")
parser.add_argument("--always-high-vram")

# Attention
parser.add_argument("--attention-split")
parser.add_argument("--attention-quad")
parser.add_argument("--attention-pytorch")

# Precision
parser.add_argument("--all-in-fp32")
parser.add_argument("--all-in-fp16")
parser.add_argument("--unet-in-fp8-e4m3fn")
```

### 2.6 مجلد presets/

**ملفات الإعدادات المسبقة:**
- `default.json`: الإعدادات الافتراضية (JuggernautXL v8)
- `anime.json`: إعدادات الأنمي
- `realistic.json`: إعدادات الواقعية
- `lightning.json`: إعدادات الأداء السريع
- `lcm.json`: إعدادات LCM

**بنية الملف:**
```json
{
    "default_model": "juggernautXL_v8Rundiffusion.safetensors",
    "default_loras": [[true, "sd_xl_offset_example-lora_1.0.safetensors", 0.1]],
    "default_cfg_scale": 4.0,
    "default_sample_sharpness": 2.0,
    "default_sampler": "dpmpp_2m_sde_gpu",
    "default_scheduler": "karras",
    "checkpoint_downloads": {...},
    "lora_downloads": {...}
}
```

### 2.7 مجلد extras/

**الإضافات والمحاضرات:**
- `expansion.py`: Fooocus Prompt Expansion
- `ip_adapter.py`: دعم IP-Adapter
- `inpaint_mask.py`: توليد القناع (SAM, U2Net)
- `preprocessors.py`: المعالجة المسبقة (Canny, CPDS)
- `face_crop.py`: قص الوجوه
- `censor.py`: فحص NSFW
- `vae_interpose.py`: تحسين VAE

### 2.8 مجلد models/

**هيكل النماذج:**
```
models/
├── checkpoints/     # نماذج SDXL
├── loras/          # ملفات LoRA
├── embeddings/     # Embeddings
├── vae/            # نماذج VAE
├── controlnet/     # نماذج ControlNet
├── inpaint/        # نماذج Inpaint
├── upscale_models/ # نماذج التكبير
├── clip_vision/    # نماذج CLIP Vision
├── sam/            # نماذج SAM
└── prompt_expansion/ # نماذج التوسيع
```

### 2.9 مجلد wildcards/

**ملفات Wildcards للنماذج النصية:**
- `animal.txt`: أسماء حيوانات
- `artist.txt`: أسماء فنانين
- `color.txt`: ألوان
- `flower.txt`: أزهار

**الاستخدام في البرومبت:**
```
__animal__ in a __color__ forest → "fox in a blue forest"
```

---

## القسم 3: إدارة الحزم وال依赖يات

### 3.1 تحليل requirements_versions.txt

**الحزم الحرجة:**

| الحزمة | الإصدار | السبب |
|--------|---------|-------|
| `torch` | 2.1.0 | المحرك الأساسي |
| `gradio` | 3.41.2 | واجهة المستخدم |
| `transformers` | 4.42.4 | معالجة النصوص |
| `safetensors` | 0.4.3 | تحميل الأوزان |
| `accelerate` | 0.32.1 | تسريع التحميل |
| `einops` | 0.8.0 | معالجة المصفوفات |
| `pillow` | 10.4.0 | معالجة الصور |

### 3.2 سبب الحساسية للمصادر

**Gradio 3.41.2:**
- Fooocus يعتمد على API داخلي غير مستقر
- التحديث يكسر واجهة `gr.Blocks`
- مشاكل في `queue()` و `event handlers`

**PyTorch 2.1.0:**
- متوافق مع xformers 0.0.23
- دعم cu121 محسّن
- `torch.cuda.amp` يعمل بشكل صحيح

**Jinja2 و MarkupSafe:**
- Gradio يستخدمهما للقوالب
- التحديث يسبب `ImportError` في `gradio.templates`

**سبب انهيار المشروع:**
```
Gradio 3.41.2 → يستخدم Jinja2 3.1.x
    ↓
 MarkupSafe 2.1.x → يكسر API قديم
    ↓
 ImportError: cannot import name 'url_quote' from 'markupsafe'
```

---

## القسم 4: دورة حياة التشغيل والتدفق الحسابي

### 4.1 المسار من `python launch.py` إلى Local URL

```
1. launch.py
   ├── 1.1 تعيين المتغيرات البيئية
   ├── 1.2 prepare_environment()
   │   ├── تثبيت PyTorch 2.1.0 + cu121
   │   ├── تثبيت xformers 0.0.23 (اختياري)
   │   └── تثبيت requirements_versions.txt
   ├── 1.3 build_launcher()
   └── 1.4 ini_args()
       └── تحليل sys.argv

2. config.py
   ├── 2.1 تحميل presets/default.json
   ├── 2.2 تحميل config.txt
   └── 2.3 تحديد المسارات

3. download_models()
   ├── تحميل VAE Approx
   ├── تحميل Fooocus Expansion
   └── تحميل Checkpoints والـ LoRAs

4. webui.py
   ├── 4.1 بناء واجهة Gradio
   ├── 4.2 ربط الأحداث
   └── 4.3 تشغيل Server
       └── Local URL: http://127.0.0.1:7865
```

### 4.2 تدفق البرومبت من الواجهة إلى الإنتاج

```
المستخدم يكتب البرومبت
        ↓
gr.Textbox (prompt) → get_task()
        ↓
AsyncTask(args=args)
        ↓
worker.async_tasks.append(task)
        ↓
[Worker Thread]
        ↓
1. تطبيق Fooocus Expansion
2. تطبيق الأنماط (Styles)
3. توليد الإحداثيات (Seed, Steps)
4. تحميل النموذج (Base Model)
5. تطبيق LoRA
6. ترميز البرومبت (CLIP)
7. إنشاء Latent Space
8. العينات (KSampler)
9. ترميز VAE
10. حفظ الصورة
        ↓
task.yields.append(['results', img_paths])
        ↓
generate_clicked() → yield gr.update()
        ↓
الصورة تظهر في Gallery
```

**الكود الرئيسي في async_worker.py:**
```python
def worker():
    while True:
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            # 1. تطبيق الأنماط
            # 2. توسيع البرومبت
            # 3. توليد الصور
            # 4. إرجاع النتائج
```

---

## القسم 5: استراتيجيات إدارة الموارد والذاكرة

### 5.1 خيارات إدارة الذاكرة

#### `--always-cpu`
**الوظيفة:** تشغيل جميع العمليات على المعالج
```python
# model_management.py:62-66
if args.always_cpu:
    if args.always_cpu > 0:
        torch.set_num_threads(args.always_cpu)
    cpu_state = CPUState.CPU
```
**التأثير:**
- لا يستخدم GPU إطلاقاً
- بطيء جداً لكن يعمل على أي جهاز
- مفيد للاختبار أو الأجهزة بدون GPU

#### `--always-low-vram`
**الوظيفة:** تفعيل وضع VRAM المنخفض
```python
# model_management.py:208-210
if args.always_low_vram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
```
**التأثير:**
- يحمّل النموذج جزئياً إلى GPU
- ينقل الأجزاء غير المستخدمة إلى CPU
- يتطلب 4GB+ VRAM

**آلية التحميل الجزئي:**
```python
# model_management.py:306-322
if lowvram_model_memory > 0:
    mem_counter = 0
    for m in self.real_model.modules():
        module_mem = module_size(m)
        if mem_counter + module_mem < lowvram_model_memory:
            m.to(self.device)
            mem_counter += module_mem
```

#### `--attention-split`
**الوظيفة:** تقسيم عمليات الانتباه لتقليل استهلاك الذاكرة
```python
# args_parser.py:91-92
attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--attention-split", action="store_true")
```
**التأثير:**
- يقسم مصفوفات الانتباه الكبيرة
- يقلل استهلاك VRAM بنسبة 30-50%
- قد يبطئ العملية قليلاً

**المقارنة مع الخيارات الأخرى:**
| الخيار | الذاكرة | السرعة |
|--------|---------|--------|
| `--attention-split` | أقل | أبطأ |
| `--attention-quad` | متوسط | متوسط |
| `--attention-pytorch` | أعلى | أسرع |

#### `--lowvram`
**الملاحظة:** هذا المعامل غير موجود في الكود المصدري. المعامل الصحيح هو `--always-low-vram`.

### 5.2 آلية إدارة الذاكرة العامة

**تسلسل تحميل النموذج:**
```python
def load_models_gpu(models, memory_required=0):
    # 1. حساب الذاكرة المطلوبة
    # 2. تحرير الذاكرة إذا لزم
    # 3. تحميل النموذج
    # 4. في وضع lowvram: تحميل جزئي
```

**تحرير الذاكرة:**
```python
def free_memory(memory_required, device, keep_loaded=[]):
    for i in range(len(current_loaded_models) -1, -1, -1):
        if get_free_memory(device) > memory_required:
            break
        # تفريغ النموذج من GPU
        current_loaded_models.pop(i).model_unload()
```

### 5.3 نصائح للتحسين

1. **لديك 8GB+ VRAM:** استخدم الإعدادات الافتراضية
2. **لديك 4-8GB VRAM:** استخدم `--always-low-vram`
3. **لديك أقل من 4GB:** استخدم `--always-cpu`
4. **لبطء الانتباه:** جرب `--attention-split`

---

## دليل المطورين: التعديل على الكود

### نقاط الدخول الرئيسية للتعديل

1. **إضافة ميزة جديدة في الواجهة:** `webui.py`
2. **تعديل خوارزمية التوليد:** `modules/async_worker.py`
3. **إضافة نموذج جديد:** `modules/config.py` + `presets/`
4. **تحسين الأداء:** `modules/patch.py` + `ldm_patched/`
5. **إضافة معامل CLI:** `ldm_patched/modules/args_parser.py`

### مخطط ال依赖يات

```
launch.py
    ↓
args_manager.py → ldm_patched/modules/args_parser.py
    ↓
modules/config.py
    ↓
webui.py → modules/async_worker.py
                ↓
        modules/default_pipeline.py → modules/core.py
                ↓
        ldm_patched/modules/model_management.py
```

---

**المطور:** lllyasviel
**الإصدار:** 2.5.5
**التاريخ:** 2024
**الترخيص:** GNU AGPL v3.0
