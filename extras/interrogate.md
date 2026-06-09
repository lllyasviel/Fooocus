---
### 📄 docs/extras/interrogate.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
توليد وصف نصي (caption) للصور باستخدام BLIP (Bootstrapping Language-Image Pre-training). يُحمّل النموذج عند الطلب (lazy loading)، يُعالج الصورة بإعادة التحجيم (384×384) والتطبيع، ثم يُولّد وصفاً طبيعياً باستخدام BLIP Decoder مع تسلسل إخراج يصل إلى 50 رمزاً.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class Interrogator**: محرك وصف الصور BLIP.
  - `__init__(self)`: تهيئة مراجع النموذج، الجهاز، وإعدادات dtype (FP16 إن أمكن).
  - `interrogate(self, image)`: تحميل BLIP Decoder عند الحاجة، معالجة الصورة (resize to 384, normalize)، توليد caption عبر model.generate مع تكرار 3 مرات عند الفشل، إرجاع النص الموصوف.
- **default_interrogator**: كائن افتراضي من Interrogator().interrogate.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: ldm_patched/modules/model_management.py, modules/model_loader.py, modules/config.py, ldm_patched/modules/model_patcher.py, extras/BLIP/models/blip.py (blip_decoder), torchvision (transforms, InterpolationMode)
- **يُستدعى بواسطة**: webui.py (دالة trigger_describe), experiments_interrogate.py
- **الأثر البرمجي**: تعطيله يُلغي ميزة Describe Image في واجهة المستخدم
