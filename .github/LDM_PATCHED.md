# مجلد ldm_patched/ - محرك ComfyUI المعدّل

## فهرس الملفات الرئيسية

| الملف | الوظيفة |
|-------|---------|
| `modules/sd.py` | تحميل النماذج |
| `modules/samplers.py` | خوارزميات العينات |
| `modules/sample.py` | تنفيذ التوليد |
| `modules/model_management.py` | إدارة الذاكرة |
| `modules/model_patcher.py` | تطبيق الـ patches |
| `modules/args_parser.py` | تحليل المعاملات |
| `modules/options.py` | خيارات التحليل |
| `modules/conds.py` | إدارة الشروط |
| `modules/controlnet.py` | تحميل ControlNet |
| `modules/clip_vision.py` | نموذج CLIP Vision |
| `modules/sd1_clip.py` | CLIP SD1.5 |
| `modules/sdxl_clip.py` | CLIP SDXL |
| `modules/model_base.py` | النماذج الأساسية |
| `modules/model_detection.py` | كشف النماذج |
| `modules/lora.py` | تحميل LoRA |
| `modules/latent_formats.py` | تنسيقات Latent |
| `modules/utils.py` | دوال مساعدة |
| `modules/ops.py` | العمليات الحسابية |

---

## 📄 ldm_patched/modules/sd.py

### 1. الوظيفة الأساسية (Core Function)
تحميل النماذج والتعامل مع ملفات الأوزان.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `ldm_patched.modules.model_management` - إدارة الذاكرة
- `ldm_patched.ldm.models.autoencoder` - VAE
- `ldm_patched.modules.model_detection` - كشف النموذج
- `ldm_patched.modules.sd1_clip`, `sd2_clip`, `sdxl_clip`

**الفئات:**
- `CLIP`: نموذج CLIP

**الدوال:**
- `load_model_weights()`: تحميل الأوزان
- `load_clip_weights()`: تحميل أوزان CLIP
- `load_lora_for_models()`: تحميل LoRA
- `load_checkpoint_guess_config()`: تحميل checkpoint مع تخمين الإعدادات

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `modules/core.py`
- **يستخدم:** `modules/model_management.py`, `modules/model_detection.py`
- **التأثير عند التعديل:** تعطيل تحميل النماذج بالكامل

---

## 📄 ldm_patched/modules/samplers.py

### 1. الوظيفة الأساسية (Core Function)
تنفيذ خوارزميات العينات (Sampling) المختلفة.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `ldm_patched.k_diffusion.sampling` - خوارزميات k-diffusion
- `ldm_patched.unipc.uni_pc` - UniPC sampler

**الفئات:**
- `KSampler`: العينات الرئيسية

**الدوال:**
- `sample()`: تنفيذ العينات
- `calc_cond_uncond_batch()`: حساب conditioning
- `get_area_and_mult()`: حساب المنطقة والمضاعف
- `encode_model_conds()`: تشفير الشروط

**الخوارزميات المدعومة:**
- euler, euler_ancestral, heun
- dpm_2, dpm_2_ancestral
- dpmpp_2s_ancestral, dpmpp_2m, dpmpp_2m_sde
- ddim, uni_pc, lcm

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `modules/sample.py`, `modules/sample_hijack.py`
- **يستخدم:** `ldm_patched/k_diffusion/sampling.py`
- **التأثير عند التعديل:** تعطيل العينات بالكامل

---

## 📄 ldm_patched/modules/sample.py

### 1. الوظيفة الأساسية (Core Function)
إعداد وتنفيذ عملية التوليد.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `ldm_patched.modules.model_management`
- `ldm_patched.modules.samplers`
- `ldm_patched.modules.conds`

**الدوال:**
- `prepare_noise()`: تحضير الضوضاء
- `prepare_mask()`: تحضير القناع
- `get_additional_models()`: الحصول على النماذج الإضافية
- `prepare_sampling()`: إعداد العينات
- `sample()`: التنفيذ الرئيسي

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `modules/core.py`
- **يستخدم:** `modules/samplers.py`, `modules/model_management.py`
- **التأثير عند التعديل:** تعطيل التوليد

---

## 📄 ldm_patched/modules/model_management.py

### 1. الوظيفة الأساسية (Core Function)
إدارة الذاكرة وتحميل النماذج على GPU.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `psutil` - معلومات الذاكرة
- `torch` - PyTorch

**الفئات:**
- `VRAMState`: حالات VRAM (DISABLED, NO_VRAM, LOW_VRAM, NORMAL_VRAM, HIGH_VRAM, SHARED)
- `CPUState`: حالات المعالج (GPU, CPU, MPS)
- `LoadedModel`: نموذج محمل

**الدوال الرئيسية:**
- `get_torch_device()`: الحصول على الجهاز المستخدم
- `get_total_memory()`: الحصول على إجمالي الذاكرة
- `load_models_gpu()`: تحميل النماذج إلى GPU
- `free_memory()`: تحرير الذاكرة
- `soft_empty_cache()`: تفريغ الكاش

**منطق التحميل:**
- يتحقق من الذاكرة المتاحة
- يحمّل النماذج حسب الأولوية
- يدعم LOW_VRAM mode

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** جميع ملفات ldm_patched تقريباً
- **يستخدم:** `modules/args_parser.py`
- **التأثير عند التعديل:** تعطيل إدارة الذاكرة بالكامل

---

## 📄 ldm_patched/modules/model_patcher.py

### 1. الوظيفة الأساسية (Core Function)
تطبيق الـ patches على النماذج دون تغيير الأوزان الأصلية.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `ModelPatcher`: مُطبّق الـ patches

**الدوال الرئيسية:**
- `clone()`: نسخ النموذج مع الـ patches
- `add_patches()`: إضافة patches
- `remove_patches()`: إزالة patches
- `set_model_patch()`: تعيين patch
- `set_model_patch_replace()`: استبدال patch
- `set_model_attn1_patch()`: تعيين attn1 patch
- `set_model_attn2_patch()`: تعيين attn2 patch

**أنواع الـ patches:**
- `attn1_patch`, `attn2_patch`
- `attn1_output_patch`, `attn2_output_patch`
- `input_blocks_patch`, `output_blocks_patch`
- `middle_block_patch`

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** جميع ملفات ldm_patched
- **يستخدم:** `modules/model_management.py`
- **التأثير عند التعديل:** تعطيل نظام الـ patches بالكامل

---

## 📄 ldm_patched/modules/args_parser.py

### 1. الوظيفة الأساسية (Core Function)
تحليل معاملات CLI الأساسية للمشروع.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `argparse`, `ldm_patched.modules.options`

**المعاملات الرئيسية:**
- `--listen`, `--port`: خادم الشبكة
- `--always-cpu`, `--always-low-vram`, `--always-no-vram`: إدارة الذاكرة
- `--attention-split`, `--attention-quad`, `--attention-pytorch`: الانتباه
- `--all-in-fp32`, `--all-in-fp16`: الدقة العامة
- `--unet-in-fp8-e4m3fn`, `--unet-in-fp8-e5m2`: دقة UNet
- `--vae-in-fp16`, `--vae-in-fp32`, `--vae-in-bf16`: دقة VAE
- `--clip-in-fp8-e4m3fn`, `--clip-in-fp16`, `--clip-in-fp32`: دقة CLIP
- `--preview-option`: خيار المعاينة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `args_manager.py`
- **يستخدم:** `modules/options.py`
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
- **يستخدمه:** `args_parser.py`
- **التأثير عند التعديل:** تغيير سلوك تحليل المعاملات

---

## 📄 ldm_patched/modules/conds.py

### 1. الوظيفة الأساسية (Core Function)
إدارة شروط conditioning للتوليد.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `CONDRegular`: شروط عادية
- `CONDCrossAttn`: شروط Cross Attention

**الدوال:**
- `convert_cond()`: تحويل الشروط

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/samplers.py`, `modules/sample.py`
- **التأثير عند التعديل:** تعطيل إدارة الشروط

---

## 📄 ldm_patched/modules/controlnet.py

### 1. الوظيفة الأساسية (Core Function)
تحميل وتطبيق ControlNet.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `load_controlnet()`: تحميل ControlNet
- `apply_controlnet()`: تطبيق ControlNet

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/core.py`
- **التأثير عند Accreditation:** تعطيل ControlNet

---

## 📄 ldm_patched/modules/clip_vision.py

### 1. الوظيفة الأساسية (Core Function)
تحميل وتشغيل نموذج CLIP Vision.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `ClipVisionModel`: نموذج CLIP Vision

**الدوال:**
- `load()`: تحميل النموذج
- `clip_preprocess()`: معالجة الصورة

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `extras/ip_adapter.py`
- **التأثير عند التعديل:** تعطيل CLIP Vision

---

## 📄 ldm_patched/modules/sd1_clip.py

### 1. الوظيفة الأساسية (Core Function)
نموذج CLIP لـ SD1.5.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `SDClipModel`: نموذج CLIP
- `ClipTokenWeightEncoder`: تشفير أوزان الرمز

**الدوال:**
- `gen_empty_tokens()`: توليد tokens فارغة

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/patch_clip.py`
- **التأثير عند التعديل:** تعطيل CLIP لـ SD1.5

---

## 📄 ldm_patched/modules/sdxl_clip.py

### 1. الوظيفة الأساسية (Core Function)
نموذج CLIP لـ SDXL.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `SDXLClipModel`: نموذج CLIP لـ SDXL

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/sd.py`
- **التأثير عند التعديل:** تعطيل CLIP لـ SDXL

---

## 📄 ldm_patched/modules/model_base.py

### 1. الوظيفة الأساسية (Core Function)
النماذج الأساسية لـ Stable Diffusion.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `SDXL`: نموذج SDXL الأساسي
- `SDXLRefiner`: نموذج SDXL Refiner
- `SD15`: نموذج SD1.5

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/sd.py`, `modules/sample_hijack.py`
- **التأثير عند التعديل:** تعطيل النماذج الأساسية

---

## 📄 ldm_patched/modules/model_detection.py

### 1. الوظيفة الأساسية (Core Function)
كشف نوع النموذج تلقائياً.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `detect_checkpoint_config()`: كشف إعدادات checkpoint

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/sd.py`
- **التأثير عند Accreditation:** تعطيل الكشف التلقائي

---

## 📄 ldm_patched/modules/lora.py

### 1. الوظيفة الأساسية (Core Function)
تحميل وإدارة ملفات LoRA.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `model_lora_keys_unet()`: الحصول على مفاتيح LoRA لـ UNet
- `model_lora_keys_clip()`: الحصول على مفاتيح LoRA لـ CLIP
- `load_lora()`: تحميل LoRA

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/sd.py`, `modules/core.py`
- **التأثير عند التعديل:** تعطيل دعم LoRA

---

## 📄 ldm_patched/modules/latent_formats.py

### 1. الوظيفة الأساسية (Core Function)
تنسيقات Latent space.

### 2. التفاصيل الفنية والبرمجية
**الفئات:**
- `LatentFormat`: التنسيق الأساسي
- `SDXL`: تنسيق SDXL
- `SD15`: تنسيق SD1.5

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/sd.py`
- **التأثير عند التعديل:** تغيير تنسيقات Latent

---

## 📄 ldm_patched/modules/utils.py

### 1. الوظيفة الأساسية (Core Function)
دوال مساعدة متنوعة.

### 2. التفاصيل الفنية والبرمجية
**الدوال:**
- `transformers_convert()`: تحويل مفاتيح transformers
- `repeat_to_batch_size()`: تكرار إلى حجم الدفعة
- `load_torch_file()`: تحميل ملف PyTorch

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** معظم ملفات ldm_patched
- **التأثير عند التعديل:** تعطيل الدوال المساعدة

---

## 📄 ldm_patched/modules/ops.py

### 1. الوظيفة الأساسية (Core Function)
العمليات الحسابية المخصصة.

### 2. التفاصيل الفنية والبرمجية
**المتغيرات:**
- `manual_cast`: التحويل اليدوي للنوع

### 3. شبكة العلاقات ودورة الحياة
- **يستخدمه:** `modules/patch_clip.py`, `extras/ip_adapter.py`
- **التأثير عند التعديل:** تغيير العمليات الحسابية

---

## مخطط العلاقات لمجلد ldm_patched/

```
modules/sd.py
    ├── modules/model_management.py
    ├── modules/model_detection.py
    ├── modules/model_patcher.py
    ├── modules/sd1_clip.py
    ├── modules/sdxl_clip.py
    ├── modules/clip_vision.py
    ├── modules/lora.py
    └── modules/latent_formats.py

modules/samplers.py
    ├── modules/model_management.py
    ├── modules/conds.py
    ├── k_diffusion/sampling.py
    └── unipc/uni_pc.py

modules/sample.py
    ├── modules/model_management.py
    ├── modules/samplers.py
    └── modules/conds.py

modules/model_patcher.py
    ├── modules/model_management.py
    └── modules/utils.py
```

---

**آخر تحديث:** 2026
**الإصدار:** 2.5.5
