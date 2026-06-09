---
### 📄 مسار الملف الافتراضي: docs/extras/safety_checker/safety_checker.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج كشف NSFW (Stable Diffusion Safety Checker) المُعاد تكييفه من HuggingFace diffusers. يكشف المحتوى غير اللائق باستخدام مسافة التشابه الكوسينية مع تضمينات المفاهيم المُتعلّمة.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **StableDiffusionSafetyChecker(PreTrainedModel)**: كاشف NSFW القائم على CLIP vision.
  - `forward(self, clip_input, images)`: فحص الصور وإرجاع نسخة مُعتمة + أعلام boolean.
- **cosine_distance(image_embeds, text_embeds)**: حساب مصفوفة التشابه الكوسينية.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: torch, transformers (CLIPVisionModel, PreTrainedModel)
- **يُستخدم بواسطة**: extras/censor.py
- **الأثر البرمجي**: النموذج الأساسي للكشف عن NSFW
