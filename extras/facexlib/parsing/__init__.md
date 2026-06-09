---
### 📄 docs/extras/facexlib/parsing/__init__.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملف تهيئة حزمة parsing في facexlib. يُصدّر دالة init_parsing_model لتحميل نموذج تجزئة الوجه (face parsing) حسب الاسم. يُوفّر واجهة موحدة لنماذج BiSeNet و Parsenet المستخدمة في تجزئة أجزاء الوجه (العيون، الأنف، الفم، إلخ).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **init_parsing_model(name, half=False, device='cuda')**: إنشاء نموذج تجزئة: تحميل BiSeNet أو Parsenet من الأوزان، إرجاع النموذج على الجهاز المطلوب.
  - يدعم: 'bisenet', 'parsenet'.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/facexlib/parsing/bisenet.py, extras/facexlib/parsing/parsenet.py, torch
- **يُستدعى بواسطة**: extras/facexlib/utils/face_restoration_helper.py (FaceRestoreHelper)
- **الأثر البرمجي**: نقطة الوصول الوحيدة لنماذج تجزئة الوجه
