# مجلد extras/ - الإضافات والمحاضرات

## فهرس الملفات

| الملف | الوظيفة |
|-------|---------|
| `censor.py` | فحص NSFW |
| `expansion.py` | توسيع البرومبت |
| `face_crop.py` | قص الوجوه |
| `interrogate.py` | وصف الصور (BLIP) |
| `wd14tagger.py` | وصف الصور الأنمي |
| `resampler.py` | نموذج Resampler |
| `preprocessors.py` | المعالجة المسبقة |
| `ip_adapter.py` | تطبيق IP-Adapter |
| `inpaint_mask.py` | توليد القناع |
| `vae_interpose.py` | تحويل VAE |

---

## 📄 extras/censor.py

### 1. الوظيفة الأساسية (Core Function)
فحص الصور للمحتوى NSFW وحجبها.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `transformers` - نموذج Safety Checker
- `modules.config` - تحميل النموذج
- `ldm_patched.modules.model_management` - إدارة الذاكرة

**الفئات:**
- `Censor`: فاحص NSFW

**الدوال:**
- `init()`: تحميل النموذج
- `censor()`: فحص وحجب الصور

**المنطق:**
- يستخدم CLIPImageProcessor لمعالجة الصور
- يستخدم StableDiffusionSafetyChecker للفحص

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل فحص NSFW

---

## 📄 extras/expansion.py

### 1. الوظيفة الأساسية (Core Function)
محرك توسيع البرومبت باستخدام GPT2.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `transformers` - نموذج GPT2
- `ldm_patched.modules.model_management` - إدارة الذاكرة

**الفئات:**
- `FooocusExpansion`: محرك التوسيع

**الدوال:**
- `__init__()`: تحميل النموذج والـ tokenizer
- `__call__()`: توسيع البرومبت
- `logits_processor()`: معالجة النتائج

**المنطق:**
- يستخدم GPT2 لتوليد كلمات إيجابية
- يقيّد الكلمات باستخدام positive.txt
- يدعم Seed العشوائي

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`, `default_pipeline.py`
- **التأثير عند التعديل:** تعطيل توسيع البرومبت

---

## 📄 extras/face_crop.py

### 1. الوظيفة الأساسية (Core Function)
قص الوجوه من الصور.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `cv2` - معالجة الصور
- `extras.facexlib` - كشف الوجوه

**الدوال:**
- `crop_image()`: قص الوجه الأول من الصورة
- `align_warp_face()`: محاذاة الوجه

**المنطق:**
- يستخدم FaceRestoreHelper لكشف الوجوه
- يستخدم landmarks لقص الوجه

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل قص الوجوه

---

## 📄 extras/interrogate.py

### 1. الوظيفة الأساسية (Core Function)
وصف الصور باستخدام نموذج BLIP.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `torchvision` - معالجة الصور
- `extras.BLIP.models.blip` - نموذج BLIP

**الفئات:**
- `Interrogator`: واصف الصور

**الدوال:**
- `interrogate()`: وصف صورة

**المنطق:**
- يحمّل نموذج BLIP من HuggingFace
- يعالج الصورة بحجم 384x384
- يولّد وصفاً نصياً للصورة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py` (via describe tab)
- **التأثير عند التعديل:** تعطيل الوصف التلقائي

---

## 📄 extras/wd14tagger.py

### 1. الوظيفة الأساسية (Core Function)
وصف الصور الأنمي باستخدام WD14 Tagger.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `onnxruntime` - تشغيل النموذج
- `csv` - قراءة التسميات

**الدوال:**
- `default_interrogator()`: وصف صورة

**المنطق:**
- يستخدم نموذج ONNX للتصنيف
- يقرأ التسميات من ملف CSV
- يدعم فلترة حسب العتبة

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py` (via describe tab)
- **التأثير عند التعديل:** تعطيل وصف الأنمي

---

## 📄 extras/resampler.py

### 1. الوظيفة الأساسية (Core Function)
نموذج Resampler لـ IP-Adapter.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `torch`, `torch.nn`

**الفئات:**
- `PerceiverAttention`: انتباه Perceiver
- `FeedForward`: طبقة Feed Forward
- `Resampler`: نموذج Resampler

**المنطق:**
- يستخدم PerceiverAttention لدمج الميزات
- يحول embeddings الصورة إلى latent space

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `ip_adapter.py`
- **التأثير عند التعديل:** تعطيل IP-Adapter

---

## 📄 extras/preprocessors.py

### 1. الوظيفة الأساسية (Core Function)
المعالجة المسبقة للصور (Canny, CPDS).

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:** `cv2`, `numpy`

**الدوال:**
- `canny_pyramid()`: Canny مع هرمي
- `cpds()`: CPDS preprocessing
- `centered_canny()`: Canny مركزي
- `norm255()`: تطبيع القيم

**المنطق:**
- يستخدم هرمي Canny لتحسين النتائج
- CPDS لتحويل الألوان إلى رمادي

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل المعالجة المسبقة

---

## 📄 extras/ip_adapter.py

### 1. الوظيفة الأساسية (Core Function)
تطبيق IP-Adapter على النموذج.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `safetensors.torch` - تحميل الأوزان
- `ldm_patched.modules.clip_vision` - CLIP Vision
- `extras.resampler` - Resampler

**الفئات:**
- `ImageProjModel`: نموذج إسقاط الصورة
- `To_KV`: طبقة Key/Value
- `IPAdapterModel`: النموذج الرئيسي

**الدوال:**
- `load_ip_adapter()`: تحميل IP-Adapter
- `preprocess()`: معالجة الصورة
- `patch_model()`: تطبيق على النموذج

**المنطق:**
- يستخدم CLIP Vision لاستخراج ميزات الصورة
- يدمج الميزات مع text conditioning

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `async_worker.py`
- **التأثير عند التعديل:** تعطيل IP-Adapter

---

## 📄 extras/inpaint_mask.py

### 1. الوظيفة الأساسية (Core Function)
توليد قناع Inpaint باستخدام SAM + GroundingDINO.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `segment_anything` - نموذج SAM
- `rembg` - إزالة الخلفية
- `extras.GroundingDINO` - كشف الأجسام

**الفئات:**
- `SAMOptions`: خيارات SAM

**الدوال:**
- `generate_mask_from_image()`: توليد القناع
- `optimize_masks()`: تحسين القناع

**المنطق:**
- يستخدم GroundingDINO لكشف الأجسام
- يستخدم SAM لتوليد القناع الدقيق
- يدعم نماذج متعددة (vit_b, vit_l, vit_h)

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `webui.py`, `async_worker.py`
- **التأثير عند التعديل:** تعطيل توليد القناع

---

## 📄 extras/vae_interpose.py

### 1. الوظيفة الأساسية (Core Function)
تحويل Latent من SDXL إلى SD1.5.

### 2. التفاصيل الفنية والبرمجية
**المكتبات المستوردة:**
- `safetensors.torch` - تحميل الأوزان
- `torch.nn` - بناء النموذج

**الفئات:**
- `ResBlock`: طبقة متبقية
- `ExtractBlock`: طبقة استخراج
- `InterposerModel`: نموذج التحويل

**الدوال:**
- `parse()`: تحويل Latent

**المنطق:**
- يستخدم نموذج CNN لتحويل Latent space
- يدعم FP16 لتحسين الأداء

### 3. شبكة العلاقات ودورة الحياة
- **يستدعيه:** `default_pipeline.py`
- **التأثير عند التعديل:** تعطيل التحويل

---

## مخطط العلاقات لمجلد extras/

```
async_worker.py
    ├── censor.py
    ├── expansion.py
    ├── face_crop.py
    ├── preprocessors.py
    ├── ip_adapter.py
    │   └── resampler.py
    └── inpaint_mask.py

webui.py
    ├── interrogate.py
    └── wd14tagger.py

default_pipeline.py
    └── vae_interpose.py
```

---

**آخر تحديث:** 2026
**الإصدار:** 2.5.5
